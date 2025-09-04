import json
import logging
from aperag.db.models import User
from aperag.exceptions import BusinessException
from aperag.schema import view_models
from aperag.service.chat_collection_service import chat_collection_service
from aperag.service.chat_document_service import chat_document_service
from aperag.service.chat_service import chat_service_global
from aperag.service.chat_title_service import chat_title_service
from aperag.service.collection_service import collection_service
from aperag.utils.audit_decorator import audit
from aperag.views.auth import UserManager, authenticate_websocket_user, get_user_manager, optional_user, required_user


from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, Response, UploadFile, WebSocket

from aperag.views.bot import router

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chats"])

@router.post("/bots/{bot_id}/chats")
@audit(resource_type="chat", api_name="CreateChat")
async def create_chat_view(request: Request, bot_id: str, user: User = Depends(required_user)) -> view_models.Chat:
    return await chat_service_global.create_chat(str(user.id), bot_id)


@router.get("/bots/{bot_id}/chats")
async def list_chats_view(
    request: Request,
    bot_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    user: User = Depends(required_user),
) -> view_models.ChatList:
    return await chat_service_global.list_chats(str(user.id), bot_id, page, page_size)


@router.get("/bots/{bot_id}/chats/{chat_id}")
async def get_chat_view(
    request: Request, bot_id: str, chat_id: str, user: User = Depends(required_user)
) -> view_models.ChatDetails:
    return await chat_service_global.get_chat(str(user.id), bot_id, chat_id)


@router.put("/bots/{bot_id}/chats/{chat_id}")
@audit(resource_type="chat", api_name="UpdateChat")
async def update_chat_view(
    request: Request,
    bot_id: str,
    chat_id: str,
    chat_in: view_models.ChatUpdate,
    user: User = Depends(required_user),
) -> view_models.Chat:
    return await chat_service_global.update_chat(str(user.id), bot_id, chat_id, chat_in)


@router.post("/bots/{bot_id}/chats/{chat_id}/messages/{message_id}")
@audit(resource_type="message", api_name="FeedbackMessage")
async def feedback_message_view(
    request: Request,
    bot_id: str,
    chat_id: str,
    message_id: str,
    feedback: view_models.Feedback,
    user: User = Depends(required_user),
):
    return await chat_service_global.feedback_message(
        str(user.id), chat_id, message_id, feedback.type, feedback.tag, feedback.message
    )


@router.websocket("/bots/{bot_id}/chats/{chat_id}/connect")
async def websocket_chat_endpoint(
    websocket: WebSocket, bot_id: str, chat_id: str, user_manager: UserManager = Depends(get_user_manager)
):
    """WebSocket endpoint for real-time chat with bots

    Supports cookie-based authentication to get user_id
    """
    # Authenticate user from WebSocket cookies
    user_id = await authenticate_websocket_user(websocket, user_manager)
    if not user_id:
        raise HTTPException(status_code=401, detail="Unauthorized")

    await chat_service_global.handle_websocket_chat(websocket, user_id, bot_id, chat_id)


@router.post("/bots/{bot_id}/chats/{chat_id}/title")
async def generate_chat_title_view(
    bot_id: str,
    chat_id: str,
    request_body: view_models.TitleGenerateRequest = view_models.TitleGenerateRequest(),
    user: User = Depends(optional_user),
) -> view_models.TitleGenerateResponse:
    try:
        title = await chat_title_service.generate_title(
            user_id=str(user.id),
            bot_id=bot_id,
            chat_id=chat_id,
            max_length=request_body.max_length,
            language=request_body.language,
            turns=request_body.turns,
        )
        return {"title": title}
    except BusinessException as be:
        raise HTTPException(status_code=400, detail={"error_code": be.error_code.name, "message": str(be)})


@router.post("/chat/completions/frontend", tags=["chats"])
async def frontend_chat_completions_view(request: Request, user: User = Depends(required_user)):
    body = await request.body()

    # Try to parse JSON first, fallback to text for backward compatibility
    try:
        data = json.loads(body.decode("utf-8"))
        message = data.get("message", "")
        files = data.get("files", [])
    except (json.JSONDecodeError, UnicodeDecodeError):
        # Fallback to text message for backward compatibility
        message = body.decode("utf-8")
        files = []

    query_params = dict(request.query_params)
    stream = query_params.get("stream", "false").lower() == "true"
    bot_id = query_params.get("bot_id", "")
    chat_id = query_params.get("chat_id", "")
    msg_id = request.headers.get("msg_id", "")

    return await chat_service_global.frontend_chat_completions(
        str(user.id), message, stream, bot_id, chat_id, msg_id, files
    )


@router.post("/chats/{chat_id}/search")
@audit(resource_type="search", api_name="SearchChatFiles")
async def search_chat_files_view(
    request: Request,
    chat_id: str,
    data: view_models.SearchRequest,
    user: User = Depends(required_user),
) -> view_models.SearchResult:
    """Search files within a specific chat using hybrid search capabilities"""
    try:
        # Get user's chat collection
        chat_collection_id = await chat_collection_service.get_user_chat_collection_id(str(user.id))
        if not chat_collection_id:
            raise HTTPException(status_code=404, detail="Chat collection not found")

        if not chat_id:
            raise HTTPException(status_code=400, detail="Chat ID is required")

        # Execute search flow using the helper method from collection_service
        items, _ = await collection_service.execute_search_flow(
            data=data,
            collection_id=chat_collection_id,
            search_user_id=str(user.id),
            chat_id=chat_id,  # Add chat_id for filtering in chat searches
            flow_name="chat_search",
            flow_title="Chat Search",
        )

        # Return search result without saving to database for chat searches
        from aperag.schema.view_models import SearchResult

        return SearchResult(
            id=None,  # No ID since not saved
            query=data.query,
            vector_search=data.vector_search,
            fulltext_search=data.fulltext_search,
            graph_search=data.graph_search,
            summary_search=data.summary_search,
            items=items,
            created=None,  # No creation time since not saved
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to search chat files: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/chats/{chat_id}/documents")
@audit(resource_type="document", api_name="UploadChatDocument")
async def upload_chat_document_view(
    request: Request,
    chat_id: str,
    file: UploadFile = File(...),
    user: User = Depends(required_user),
) -> view_models.Document:
    """Upload a document to a chat session"""
    return await chat_document_service.upload_chat_document(chat_id=chat_id, user_id=str(user.id), file=file)


@router.get("/chats/{chat_id}/documents/{document_id}")
async def get_chat_document_view(
    request: Request,
    chat_id: str,
    document_id: str,
    user: User = Depends(required_user),
) -> view_models.Document:
    """Get chat document details"""
    document = await chat_document_service.get_chat_document_by_id(
        chat_id=chat_id, document_id=document_id, user_id=str(user.id)
    )

    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    return document


@router.delete("/bots/{bot_id}/chats/{chat_id}")
@audit(resource_type="chat", api_name="DeleteChat")
async def delete_chat_view(request: Request, bot_id: str, chat_id: str, user: User = Depends(required_user)):
    await chat_service_global.delete_chat(str(user.id), bot_id, chat_id)
    return Response(status_code=204)