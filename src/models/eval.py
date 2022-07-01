from numpy import ndim
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, cohen_kappa_score, log_loss, matthews_corrcoef
from .model import MultiOutputModel
from .nets import DeepAutoencoder

ACCURACY = 'Accuracy'
RECALL = 'Recall'
F1 = 'F1'
ROC_AUC = 'ROC AUC'
COHEN_KAPPA = 'Cohen Kappa'
LOG_LOSS = 'Log loss'
MATTHEWS = 'Matthews CC'

single_output_measures = {
    ACCURACY : (accuracy_score, {}),
    RECALL : (recall_score, {"average" : "weighted"}),
    F1 : (f1_score, {"average" : "weighted"}),
    COHEN_KAPPA : (cohen_kappa_score, {}),
    MATTHEWS : (matthews_corrcoef, {})
}

multiple_output_measures = {
    ROC_AUC : (roc_auc_score, { "average" : "weighted", "multi_class" : "ovr" }),
    LOG_LOSS : (log_loss, {})
}

all_measures = single_output_measures.copy()
all_measures.update(multiple_output_measures)


def eval_model(model_class, dataset, params=None, ndims=None):
    results = {measure : 0 for measure in single_output_measures}
    if issubclass(model_class, MultiOutputModel):
        results.update({measure : 0 for measure in multiple_output_measures})

    folds = 0

    for x_train, y_train, x_validate, y_validate in dataset:
        print(f"Running iteration {folds+1}...")
        if params:
            model = model_class(**params)
        else:
            model =model_class()
        
        if ndims:
            print("Training autoencoder...")
            autoencoder = DeepAutoencoder(indim=44, outdim=ndims)
            autoencoder.train(x_train, epochs=4)
            x_train = autoencoder.encode(x_train)
            x_validate = autoencoder.encode(x_validate)
        
        print("Training model...")
        model.train(x_train, y_train)
        

        y_predict = model.predict(x_validate)
        if issubclass(model_class, MultiOutputModel):
            y_prob_predict = model.prob_predict(x_validate)
            
        print("Evaluating metrics...")
        for measure in results:
            method, params = all_measures[measure]
            y = y_predict if measure in single_output_measures else y_prob_predict
            results[measure] += method(y_validate, y, **params)
        
        folds += 1
    
    results = { measure : result / folds for measure, result in results.items() }

    return results