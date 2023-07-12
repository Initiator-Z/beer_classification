from pydantic import BaseModel, Field
from typing import List

class Input(BaseModel):
    brewery_name: str = Field(description='Name of the brewery')
    review_overall: float = Field(description='Overall review score, scored out of 5')
    review_aroma: float = Field(description='Aroma review score, scored out of 5')
    review_appearance: float = Field(description='Appearance review score, scored out of 5')
    review_profilename: str = Field(description='Profile name of the reviewer')
    review_palate: float = Field(description='Palate review score, scored out of 5')
    review_taste: float = Field(description='Taste review score, scored out of 5')
    beer_abv: float = Field(description='Alcohol by volume')
    review_year: int = Field(description='Year the beer was reviewed, as integer')
    review_month: int = Field(description='Month the beer was reviewed, as integer')

class Inputs(BaseModel):
    input_list: List[Input]

    def dict_inputs(self):
        return [input.dict() for input in self.input_list]