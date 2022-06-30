from sklearn.model_selection import GridSearchCV

def optimize(model, params, data, cv_, scoring_, n_jobs_):
    results = { param : { value : 0 for value in values } for param, values in params.items() }         

    for x_train, y_train, x_valid, y_valid in data:
        grid_search_model = GridSearchCV(
        estimator=model,
        param_grid=params,
        scoring = scoring_,
        n_jobs = n_jobs_,
        cv = cv_
        )
        grid_search_model.fit(x_train,y_train)
        for param, value in grid_search_model.best_params_.items():
            results[param][value] += 1

    best_params = {}
    for param, values_votes in results.items():
        votes = sorted(values_votes.items(), key=lambda x: x[1], reverse=True)
        best_value = votes[0][0]
        best_params[param] = best_value
    return best_params
     
    
    