from pydantic import BaseModel

class casePredictor(BaseModel):
    dateDecision: int
    term: int
    respondent: int
    caseOrigin: int
    issue: int