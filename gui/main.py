from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import os
import pandas as pd

# Get the directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
RESULTS_HTML_PATH = os.path.join(BASE_DIR, "results.html")
RESULTS_CSV_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "experiments", "output", "v3", "results.csv"))

# This is a simplified version of the model order from impossible_utils.py
MODEL_ORDER = [
    "english",
    "reverse_control",
    "reverse_full",
    "shuffle_local3",
    "shuffle_local5",
    "shuffle_local10",
    "shuffle_even_odd",
    "reverse_partial",
    "shuffle_deterministic21",
    "shuffle_deterministic57",
    "shuffle_deterministic84",
    "shuffle_nondeterministic",
]

app = FastAPI(title="Linguistic Experiments Viewer")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the results HTML page"""
    return FileResponse(RESULTS_HTML_PATH)

@app.get("/api/results")
async def get_results():
    """API endpoint to get experiment results data"""
    try:
        if not os.path.exists(RESULTS_CSV_PATH):
            raise HTTPException(status_code=404, detail=f"results.csv not found at {RESULTS_CSV_PATH}")
        df = pd.read_csv(RESULTS_CSV_PATH)
        # rename columns to be more JS friendly
        df.columns = [col.strip().replace(' ', '_') for col in df.columns]
        results = df.to_dict('records')
        return {"results": results, "model_order": MODEL_ORDER}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)