import asyncio

from aperag.db.nebula_sync_manager import NebulaSyncConnectionManager

# Initialize connection manager first
NebulaSyncConnectionManager.initialize()
NebulaSyncConnectionManager.prepare_space("foobar")


def _sync_has_node(node_id: str):
    with NebulaSyncConnectionManager.get_session("foobar") as session:
        # Use MATCH syntax - Nebula supports both nGQL and Cypher-like syntax!
        from nebula3.common import ttypes
        param_value = ttypes.Value()
        param_value.set_sVal(node_id)
        params = {"vid": param_value}
        
        query = "MATCH (v) WHERE id(v) == $vid RETURN v LIMIT 1"
        result = session.execute_parameter(query, params)
        return result.is_succeeded() and result.row_size() > 0

b = _sync_has_node("foobar")
print(b)

# with NebulaSyncConnectionManager.get_session(space="foobar") as session:
#     # 使用 execute() 而不是 execute_json()
#     result = session.execute("INVALID QUERY")
#     print(f"Result type: {type(result)}")
#     print(f"Is succeeded: {result.is_succeeded()}")
#     print(f"Error code: {result.error_code()}")
#     print(f"Error message type: {type(result.error_msg())}")
#     print(f"Error message: {result.error_msg()}")