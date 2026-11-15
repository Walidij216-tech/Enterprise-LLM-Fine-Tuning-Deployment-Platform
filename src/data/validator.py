from pydantic import BaseModel


class ProductOutput(BaseModel):
    name: str
    price: str
    category: str
    manufacturer: str