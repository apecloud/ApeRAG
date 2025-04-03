#!/usr/bin/env sh

set -e

function info() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

if [ -z "${KB_DOC_SRC_BRANCH}" ]; then
    KB_DOC_SRC_BRANCH=main
fi

if [ -z "${KB_DOC_DST_BRANCH}" ]; then
    KB_DOC_DST_BRANCH=aperag-kubeblocks-docs-${KB_DOC_SRC_BRANCH}
fi

if [ -z "${KC_API_KEY}"]; then
  info "env KC_API_KEY can not be empty"
  exit 1
fi

if [ -z "${KC_BOT_ID}" ]; then
  info "env KC_BOT_ID can not be empty"
  exit 1
fi

KB_DIR=/root/kubeblocks
KB_DOC_DIR=${KB_DIR}/docs
KC_COLLECTION_PREFIX=kubeblocks-daily-docs
KC_NEW_COLLECTION_NAME=${KC_COLLECTION_PREFIX}-$(date +%Y%m%d)
KC_OLD_COLLECTION_NAME=${KC_COLLECTION_PREFIX}-$(date -d "yesterday" +"%Y%m%d")
KC_CREATE_COLLECTION_DATA='{"config":"{\"source\":\"github\",\"crontab\":{\"enabled\":false,\"minute\":\"0\",\"hour\":\"0\",\"day_of_month\":\"*\",\"month\":\"*\",\"day_of_week\":\"*\"},\"github\":{\"repo\":\"https://github.com/apecloud/kubeblocks\",\"branch\":\"KB_DOC_DST_BRANCH\",\"path\":\"docs\"}}","title":"KC_NEW_COLLECTION_NAME","type":"document"}'
KC_CREATE_COLLECTION_DATA_FILE=/tmp/KC_CREATE_COLLECTION_DATA
echo $KC_CREATE_COLLECTION_DATA > ${KC_CREATE_COLLECTION_DATA_FILE}
sed -i "s/KB_DOC_DST_BRANCH/${KB_DOC_DST_BRANCH}/" ${KC_CREATE_COLLECTION_DATA_FILE}
sed -i "s/KC_NEW_COLLECTION_NAME/${KC_NEW_COLLECTION_NAME}/" ${KC_CREATE_COLLECTION_DATA_FILE}
KC_CURL_OPTIONS=(
  -k
  --fail
  -H "Authorization: api-key ${KC_API_KEY}"
)
KC_CURL_ENDPOINT=https://chat.kubeblocks.io

if [ ! -d "${KB_DIR}" ]; then
    git clone https://github.com/apecloud/kubeblocks.git ${KB_DIR}
fi

cd ${KB_DIR}

info "Stash changes"
git stash

info "Checkout main"
git checkout main

info "Checkout ${KB_DOC_SRC_BRANCH}"
git checkout ${KB_DOC_SRC_BRANCH}

info "Pull latest changes of branch ${KB_DOC_SRC_BRANCH}"
git pull

if git branch -r | grep ${KB_DOC_DST_BRANCH}; then
  info "Delete remote branch ${KB_DOC_DST_BRANCH}"
  git push -d origin ${KB_DOC_DST_BRANCH}
fi

if git branch | grep ${KB_DOC_DST_BRANCH}; then
  git branch -D ${KB_DOC_DST_BRANCH}
fi

git checkout -b ${KB_DOC_DST_BRANCH}

cd ${KB_DOC_DIR}

info "Delete all non-markdown files"
find . -type f ! -name "*.md" -delete

info "Show changes"
git status

info "Commit changes"
git commit -am 'chore: update docs' 2>&1

info "Push changes"
git push origin ${KB_DOC_DST_BRANCH}

while IFS=$'\t' read -r id title; do
  if [ "${KC_NEW_COLLECTION_NAME}" == "${title}" ]; then
    new_collection_id=${id}
  fi
  if [ "${KC_OLD_COLLECTION_NAME}" == "${title}" ]; then
    old_collection_id=${id}
  fi
done < <(curl -s "${KC_CURL_OPTIONS[@]}" "${KC_CURL_ENDPOINT}/api/v1/collections" | jq -r '.data[] | [.id, .title] | @tsv')

if [ -z "${new_collection_id}" ]; then
  info "Create collection ${KC_NEW_COLLECTION_NAME}"
  set +e
  stdout=`curl -s -X POST "${KC_CURL_OPTIONS[@]}" -d@${KC_CREATE_COLLECTION_DATA_FILE} ${KC_CURL_ENDPOINT}/api/v1/collections`
  set -e
  new_collection_id=`echo $stdout | jq -r .data.id`
fi
rm -rf ${KC_CREATE_COLLECTION_DATA_FILE}

if [ -z "${new_collection_id}" ]; then
  info "Can not find collection with title ${KC_NEW_COLLECTION_NAME}"
  exit 1
fi

while true; do
  status=`curl -s "${KC_CURL_OPTIONS[@]}" ${KC_CURL_ENDPOINT}/api/v1/collections/${new_collection_id} | jq -r '.data.status'`
  if [ "${status}" == "ACTIVE" ]; then
    info "Collection ${KC_NEW_COLLECTION_NAME} is active now"
    break
  fi
  info "Waiting for collection ${KC_NEW_COLLECTION_NAME} to be active"
  sleep 5
done

while true; do
  pending_docs=`curl -s "${KC_CURL_OPTIONS[@]}" ${KC_CURL_ENDPOINT}/api/v1/collections/${new_collection_id}/documents | jq -r '.data[] | select(.status != "COMPLETE") | .id' | wc -l`
  if [ "${pending_docs}" == "0" ]; then
    break
  fi
  info "Waiting for ${pending_docs} docs to be completed"
  sleep 5
done

KC_UPDATE_BOT_DATA='{"id":"KC_BOT_ID","type":"knowledge","collection_ids":["KC_NEW_COLLECTION_ID"],"title":"kubeblocks.io bot","description":null,"config":"{\"model\":\"azure-openai\",\"llm\":{\"prompt_template\":\"Candidate answers are as follows:\\n----------------\\n{context}\\n----------------\\n\\nYou are an expert in answering questions. You can draw on your own knowledge, candidate answers, and the recent conversation.\\n\\nYou need to synthesize all the information to answer the question: {query}.\\n\\nYour answer must strictly adhere to the following rules:\\n* Other than code and specific names and citations, you must speak in the same language as the question.\\n* Please think step by step to ensure accuracy and conciseness in your response.\\n* If the relevant information is not in the candidate answers, consider your own knowledge.\\n* If your own knowledge does not provide an answer, you need to inform the user that the question is beyond the scope.\\n* DO NOT repeat your self.\\n* You can only answer questions related to areas such as kubeblocks, databases, Kubernetes, operating systems, etc. If the question is unrelated to these areas, you should output a welcome message.\\n\\nThe welcome message is as follows:\\n\\nHello, I am KB Assistant!\\n\\nAs your intelligent companion, I can chat with you and help answer your questions related to areas such as kubeblocks, databases, Kubernetes, operating systems, etc.\\n\",\"context_window\":16384,\"similarity_score_threshold\":0.5,\"similarity_topk\":10,\"temperature\":0,\"enable_keyword_recall\":false,\"deployment_id\":\"\",\"api_version\":\"\",\"endpoint\":\"\",\"token\":\"\",\"trial\":true},\"use_related_question\":false,\"memory\":false}"}'
KC_UPDATE_BOT_DATA_FILE=/tmp/KC_UPDATE_BOT_DATA
echo ${KC_UPDATE_BOT_DATA} > ${KC_UPDATE_BOT_DATA_FILE}
sed -i "s/KC_BOT_ID/${KC_BOT_ID}/" ${KC_UPDATE_BOT_DATA_FILE}
sed -i "s/KC_NEW_COLLECTION_ID/${new_collection_id}/" ${KC_UPDATE_BOT_DATA_FILE}
set +e
curl -s -X PUT "${KC_CURL_OPTIONS[@]}" -d@$KC_UPDATE_BOT_DATA_FILE ${KC_CURL_ENDPOINT}/api/v1/bots/${KC_BOT_ID}
if [ $? != 0 ]; then
  info "Update bot config failed"
  rm -rf ${KC_UPDATE_BOT_DATA_FILE}
  exit 1
else
  rm -rf ${KC_UPDATE_BOT_DATA_FILE}
  info "Successfully update bot to use latest collection ${KC_NEW_COLLECTION_NAME}"
fi
set -e

if [ -n "${old_collection_id}" ]; then
  info "Delete last collection ${KC_OLD_COLLECTION_NAME}"
  curl -s -X DELETE "${KC_CURL_OPTIONS[@]}" ${KC_CURL_ENDPOINT}/api/v1/collections/${old_collection_id}
fi