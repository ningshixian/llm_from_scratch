import evaluate

index = evaluate.list_evaluation_modules(
    module_type="comparison",   # one of 'metric', 'comparison', or 'measurement'
    include_community=False,
    with_details=True
)
print(index)


# precision_metric = evaluate.load("precision")
# print(precision_metric.description)
# results = precision_metric.compute(references=[0, 1], predictions=[0, 1])
# print(results)

# clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
# results = clf_metrics.compute(predictions=[0, 1, 0], references=[0, 1, 1])
# print(results)
