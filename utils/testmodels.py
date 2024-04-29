from typing import List, Dict
from sklearn import metrics as skm
import matplotlib.pyplot as plt


def test_models(
    model_list: List[Dict],
    X_train,
    X_test,
    y_train,
    y_test,
):
    for mdl in model_list:
        model = mdl.get("estimator")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        probas = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = skm.roc_curve(y_test, probas)

        model_name = mdl.get("model_name")
        accuracy = skm.accuracy_score(
            y_test,
            y_pred,
        )
        precision = skm.precision_score(
            y_true=y_test,
            y_pred=y_pred,
            average="weighted",
        )
        recall = skm.recall_score(
            y_true=y_test,
            y_pred=y_pred,
            average="weighted",
        )
        f1_score = skm.f1_score(
            y_true=y_test,
            y_pred=y_pred,
            average="weighted",
        )
        auc = skm.roc_auc_score(
            y_true=y_test,
            y_pred=y_pred,
        )

        plt.plot(
            fpr,
            tpr,
            label="%s ROC (AUC = %0.2f)" % (mdl.get("model_name"), auc),
        )
        print(
            f"""
            ===================
            Model: {model_name}
            Accuracy: {accuracy}
            Precision: {precision}
            Recall: {recall}
            F1 Score: {f1_score}
            ROC-AUC: {auc}
            ====================
            """
        )

    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-AUC Curve")
    plt.legend(loc="lower right")
    plt.show()
