from sklearn.model_selection import GridSearchCV

def optimize(model, params, x_train, y_train, cv_, scoring_, n_jobs_):
    
    grid_search_model = GridSearchCV(
    estimator=model,
    param_grid=params,
    scoring = scoring_,
    n_jobs = n_jobs_,
    cv = cv_
    )
    grid_search_model.fit(x_train,y_train)
    print(grid_search_model.best_params_)
    #print(grid_search_model.best_estimator_)
     
    
    