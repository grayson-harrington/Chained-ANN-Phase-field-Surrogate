import pickle
import h5py
import numpy as np

from NN_classifier_main import build_classifier
from NN_regressor_main import build_regressor

from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, classification_report, confusion_matrix

import matplotlib.pyplot as plt


def cv_chained_ann(parameters, homonohomo, correlations, n_folds=1, random_seed=0, save_tests=False,
                   classifier_kwargs=None, regressor_kwargs=None):
    # Prepare k-fold data splits
    if n_folds >= 2:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
        cv_splits = kf.split(parameters)
    else:
        inds = np.arange(len(parameters))
        np.random.seed(random_seed)
        np.random.shuffle(inds)
        cutoff = int(len(inds) * 0.80)
        cv_splits = [(inds[:cutoff], inds[cutoff:])]

    # loop through cv_splits and build/test model(s)
    cv_tst_nmaes = []
    cv_tst_nstds = []
    cv_tst_r2s = []

    split_index = 0
    for train_indices, test_indices in cv_splits:
        split_index = split_index + 1

        print(f"\nCV Split: {split_index}\n")

        ### CLASSIFICATION MODEL
        print("building classifier model")
        classifier_mlp = build_classifier(parameters, homonohomo, train_indices, test_indices, **classifier_kwargs)

        print("getting heterogeneous samples for regression model building")
        hetero = np.where(homonohomo == 1)[0]
        train_indices_hetero = np.intersect1d(hetero, train_indices)
        test_indices_hetero = np.intersect1d(hetero, test_indices)

        ### DO PCA
        print("doing PCA and prepping the train/test split for regression model")
        # need to get PC scores for both train and test sets
        corrs_train = correlations[train_indices_hetero]
        corrs_test = correlations[test_indices_hetero]

        pca = PCA(
            svd_solver='full',
            n_components=5,
            random_state=random_seed,
        )
        scores_train = pca.fit_transform(corrs_train.reshape(len(corrs_train), -1))
        scores_test = pca.transform(corrs_test.reshape(len(corrs_test), -1))

        # Use PC scores to train and test model
        pc_scores = np.zeros((len(parameters), 5))
        pc_scores[train_indices_hetero] = scores_train
        pc_scores[test_indices_hetero] = scores_test

        ### REGRESSION MODEL
        print("building regression model")
        regression_mlp = build_regressor(parameters, pc_scores, train_indices_hetero, test_indices_hetero,
                                         **regressor_kwargs)

        ### CHAINED-ANN
        print("quantifying full chained-ann accuracy")
        print("predicting heterogeneous for test set")
        homonohomo_pred = classifier_mlp.predict(parameters[test_indices], scale_input=True)

        print()
        print("-" * 25)
        print()
        print(classification_report(homonohomo[test_indices], homonohomo_pred, digits=3))
        print()
        print(confusion_matrix(homonohomo[test_indices], homonohomo_pred))

        print("getting heterogeneous samples for regression from test set")
        hetero = np.where(homonohomo_pred == 1)[0]
        indices_hetero = test_indices[hetero]
        params_hetero = parameters[indices_hetero]
        corrs_hetero_true = correlations[indices_hetero]

        # transform corrs to pc_sores
        scores_hetero_true = pca.transform(corrs_hetero_true.reshape(len(corrs_hetero_true), -1))

        print("making pc score predictions")
        scores_hetero_pred = regression_mlp.predict(params_hetero, scale_input=True, unscale_output=True)

        _, nmae_tst, nstd_tst, r2_tst = regression_mlp.nMAE_nSTD_r2(scores_hetero_true, scores_hetero_pred,
                                                                    print_metrics=True)

        cv_tst_nmaes.append(nmae_tst)
        cv_tst_nstds.append(nstd_tst)
        cv_tst_r2s.append(r2_tst)

        # transform predicted pc scores back to autocorrelations
        print("reconstructing autocorrelations from PC scores")
        corrs_hetero_pred = []
        for scores in scores_hetero_pred:
            corr = np.dot(scores, pca.components_) + pca.mean_
            corr = corr.reshape(corrs_test[0].shape)
            corrs_hetero_pred.append(corr)

        if save_tests:
            print("saving results")
            with h5py.File(f"_PS Linkage/{n_folds}-fold chained-ann results.hdf5", 'a') as f:
                grp = f.create_group(f"cv{split_index}")

                grp_classifier = grp.create_group("classifier")
                grp_classifier.create_dataset("parameters", data=parameters, compression="gzip")
                grp_classifier.create_dataset("homonohomo_true", data=homonohomo, compression="gzip")
                grp_classifier.create_dataset("homonohomo_pred", data=homonohomo_pred, compression="gzip")

                grp_regressor = grp.create_group("regressor")
                grp_regressor.create_dataset("parameters", data=params_hetero, compression="gzip")
                grp_regressor.create_dataset("scores_true", data=scores_hetero_true, compression="gzip")
                grp_regressor.create_dataset("scores_pred", data=scores_hetero_pred, compression="gzip")
                grp_regressor.create_dataset("corrs_true", data=corrs_hetero_true, compression="gzip")
                grp_regressor.create_dataset("corrs_pred", data=corrs_hetero_pred, compression="gzip")

    print("\n" * 2)
    print("CV TEST ACCURACY REPORT:")
    print(f"\tnumber of folds:\t\t{n_folds}")
    print(f"\tmean nmae by PC score:\t{np.mean(cv_tst_nmaes, axis=0)}")
    print(f"\tmean nstd by PC score:\t{np.mean(cv_tst_nstds, axis=0)}")
    print(f"\tmean r2 by PC score:\t{np.mean(cv_tst_r2s, axis=0)}")
    print("\n" * 2)


dataset_path = "_datasets/dataset.hdf5"

regressor_epochs = 200  # 200 for paper
classifier_epochs = 50  # 50 for paper
cv_folds = 10  # 10 for paper
rnd_seed = 0
plot_progress = False
print_metrics = True
save_cv_tests = True

if __name__ == "__main__":
    # load in data
    with h5py.File(dataset_path, "r") as f:
        print(list(f.keys()))
        parameters = f["parameters"][...][:, 0:18]
        homonohomo = f["n_phases"][...] - 1
        correlations = f["correlations"][...]

    # perform k-fold cv for the full chained-ann
    cv_chained_ann(parameters, homonohomo, correlations, n_folds=cv_folds, random_seed=rnd_seed,
                   save_tests=save_cv_tests,
                   classifier_kwargs={"epochs": classifier_epochs,
                                      "plot_loss_error": plot_progress,
                                      "print_metrics": print_metrics},
                   regressor_kwargs={"epochs": regressor_epochs,
                                     "plot_loss_error": plot_progress,
                                     "print_metrics": print_metrics})

