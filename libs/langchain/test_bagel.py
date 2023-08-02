from langchain.vectorstores import Bagel
from bagel.config import Settings

def test_create_bagel() -> None:
    setting = Settings(
        bagel_api_impl="rest",
        bagel_server_host="api.bageldb.ai",
    )
    bagel_obj = Bagel(client_settings=setting)
    # print("ping :", bagel_obj._client.ping())


test_create_bagel()
