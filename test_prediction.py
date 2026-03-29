from src.Components.pipeline.predict_pipline import PredictPipeline

pipeline = PredictPipeline()

result = pipeline.predict([
    "fever",
    "cough",
    "headache"
])

for res in result:
    print(f"{res['disease']} → {res['probability']*100:.2f}%")