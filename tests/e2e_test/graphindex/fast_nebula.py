from aperag.db.nebula_sync_manager import NebulaSyncConnectionManager

NebulaSyncConnectionManager.prepare_space("foobar")

with NebulaSyncConnectionManager.get_session(space="foobar") as session:
    # 使用 execute() 而不是 execute_json()
    result = session.execute("INVALID QUERY")
    print(f"Result type: {type(result)}")
    print(f"Is succeeded: {result.is_succeeded()}")
    print(f"Error code: {result.error_code()}")
    print(f"Error message type: {type(result.error_msg())}")
    print(f"Error message: {result.error_msg()}")