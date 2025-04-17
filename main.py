from src.models import performance as p
from src import models as m

model = m.BClassifier()
y_test, y_pred = model.get_train_pred()
model.save()
metrics = p.eval_perform(y_pred, y_test)
p.print_perform(metrics)
p.plot_cm(metrics)

model = m.RegressionModel()
y_test, y_pred = model.get_train_pred()
model.save()
metrics = p.eval_perform(y_pred, y_test)
p.print_perform(metrics)
p.plot_cm(metrics)

