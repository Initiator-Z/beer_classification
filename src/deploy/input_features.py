from pydantic import BaseModel, Field
from typing import List

class Input(BaseModel):
    review_time: int = Field(description='Review time')
    review_overall: float = Field(description='Overall review score, scored out of 5')
    review_aroma: float = Field(description='Aroma review score, scored out of 5')
    review_appearance: float = Field(description='Appearance review score, scored out of 5')
    review_palate: float = Field(description='Palate review score, scored out of 5')
    review_taste: float = Field(description='Taste review score, scored out of 5')
    beer_abv: float = Field(description='Alcohol by volume')

class Inputs(BaseModel):
    input_list: List[Input]

    def dict_inputs(self):
        return [input.model_dump() for input in self.input_list]