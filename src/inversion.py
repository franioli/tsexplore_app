"""
Extraction of Velocity Time Series With an Optimal Temporal Sampling From
    Displacement Observation Networks - Charrier et al, TGRS, 2022

Fusion of multi-temporal and multi-sensor ice velocity observations -
    Charrier et al, ISPRS-ANNALS, 2022
"""

import datetime
import glob
import os
import pickle
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm


def GLS_solver(y, A, weight=None, regularisation=None, l=1):
    """
    General least square solver
    y is the nxn matrix of observation
    A is the nxp design matrix
    weight is the nxn weight diagonal matrix
    regularisation is the pxp regularisation diagonal matrix
    l is the scaling parameter
    """
    if weight is None:
        W = np.diag(np.ones(y.shape))
    else:
        W = weight
    if regularisation is None:
        X = np.linalg.inv(A.T @ W @ A) @ A.T @ W @ y
    else:
        L = regularisation
        # -calcolo risultato GLS
        X = np.linalg.inv(A.T @ W @ A + l * (L.T @ L)) @ A.T @ W @ y
    return X


def hat_matrix_pure(A, W):
    H = np.diag(A @ np.linalg.inv(A.T @ W @ A) @ A.T @ W)
    return H


def hat_matrix(A, W, L, l):
    H = np.diag(A @ np.linalg.inv(A.T @ W @ A + l * (L.T @ L)) @ A.T @ W)
    return H


def weight_update(X, y, A, weight=None, regularisation=None, l=1, W0=None):
    """
    update weight matrix according to Tukey0s biweight function
    X is the result matrix
    y is the nxn matrix of observation
    A is the nxp design matrix
    weight is the nxn weight diagonal matrix
    regularisation is the pxp regularisation diagonal matrix
    l is the scaling parameter
    """
    # -observations
    n = A.shape[0]
    # -numero incognite
    p = A.shape[1]
    if weight is None:
        W = np.diag(np.ones(y.shape))
    else:
        W = weight
    if W0 is None:
        W0 = np.diag(np.ones(y.shape))
    if regularisation is None:
        # H = np.diag(A @ np.linalg.inv(A.T @ W @ A) @ A.T @ W)
        H = hat_matrix_pure(A, W)
    else:
        L = regularisation
        # leverage da hat matrix
        H = hat_matrix(A, W, L, l)
        # H = np.diag(A @ np.linalg.inv(A.T @ W @ A + l*L.T @ L) @ A.T @ W)
    # -calcolo residui
    Ru = np.abs((A @ X) - y)
    # -standard deviation dei residui
    sigma = (np.sum((Ru**2) / (n - p))) ** 0.5
    # -studentized residulas
    Zu = Ru / (sigma * (1 - H) ** 0.5)
    # -Tukey's biweight funtion
    const = 4.685  # constant term
    z = Zu / np.diag(W0 + 0.001)  # add a small value to avoid divide by zeor
    # -pongo z maggiore di const per i W0 =
    # z[np.diag(W0)==0] = 5
    psi = np.zeros(len(z))
    mask = np.abs(z) < const
    psi[mask] = (1 - (Zu[mask] / const) ** 2) ** 2
    # -update weight matrix
    Wu = np.diag(psi)
    Wu = Wu * 1 / (np.mean(np.diag(Wu)))
    return Wu, Ru


def weight_definition(
    ew,  # east-west component
    ns,  # north-south comnponent
    method="residuals",  # residuls / uniform / Charrier / variable
    deltat=None,  # delta tra le osservazioni
    EW=None,  # east-west voxels
    NS=None,  # north-south voxels
    c=None,  # i-j coordinates
    cc=None,  # variable used as initial weight
):
    if method == "uniform":
        # -Pesi uniformi
        W0 = np.diag(np.ones(len(ew)))
    elif method == "Charrier":
        ii, jj = c[0], c[1]
        Iew = EW[ii - 1 : ii + 2, jj - 1 : jj + 2, :].squeeze() / deltat.squeeze()
        Ins = NS[ii - 1 : ii + 2, jj - 1 : jj + 2, :].squeeze() / deltat.squeeze()
        # -calcolo Zscore delle componenti rispetto
        tew = np.nanmedian(Iew)
        tns = np.nanmedian(Ins)
        Zew = np.abs(ew - tew) / np.median(np.abs(ew - tew))
        Zns = np.abs(ns - tns) / np.median(np.abs(ns - tns))
        # -cappo a mad=1
        mad = 1 / (Zew * Zns + 0.004)
        mad = mad / np.max(mad)
        # mad[mad>3] = 3
        # -calcolo prodotto scalare
        MA = ((ew + ns * 1j) * np.conj(np.median(Iew + Ins * 1j))) / (
            np.abs(ew + ns * 1j) * np.abs(np.median(Iew + Ins * 1j))
        )
        MA = MA.real  # * 3
        MA[MA < 0] = 0
        # -Pesi con prodotto termini di similarità
        weights = mad * MA
        W0 = np.diag(weights) / np.sum(weights).astype("float32")
    elif method == "residuals":
        # -Pesi basati sui residui
        y = ew.reshape(-1, 1) / deltat.reshape(-1, 1)
        # -provo a usarela media invece della mediana --> MOLTO MEGLIO!
        # mad1 = np.abs(y - np.median(y)) / np.median(np.abs(y-np.median(y)))
        mad1 = np.abs(y - np.mean(y)) / np.mean(np.abs(y - np.mean(y)))
        y = ns.reshape(-1, 1) / deltat.reshape(-1, 1)
        # mad2 = np.abs(y - np.median(y)) / np.median(np.abs(y-np.median(y)))
        mad2 = np.abs(y - np.mean(y)) / np.mean(np.abs(y - np.mean(y)))
        epsilon = 10e-4
        mad = 1 / (mad1 * mad2 + epsilon)
        # -metto capping a mad=3
        # mad[mad>3] = 3
        # -correggo eventuali inf
        mad[np.isinf(mad)] = np.max(mad[~np.isinf(mad)])
        W0 = np.diag(mad.squeeze()) / np.mean(mad)
        # huber = HuberRegressor().fit(deltat.reshape(-1,1), )
        # Wew = huber.predict(deltat.reshape(-1,1))
    elif method == "variable":
        # -Pesi con correlazione
        W0 = np.diag(cc) / np.mean(cc).astype("float32")

    return W0


def run(source, regularization=10, iterates=10):
    # -leggo dati
    # source = Path(source)
    data = sorted(glob.glob(f"{source}day*.txt"))
    # -leggo il primo file per inizializzare le matrici delle time series
    meta = np.loadtxt(data[0], delimiter=",")
    # -meta è matrice nx5 dove le prime due colonne sono le coordinate, la terza colonna è DX, la quarta è DY e la quinta V
    # -inizializzo i voxeloni delle componenti di spostamento
    EW = np.zeros((meta.shape[0], len(data)), dtype="float32")
    NS = np.zeros((meta.shape[0], len(data)), dtype="float32")
    # -inizializzo il voxelone del peso inziale
    WEIGHT = np.zeros((meta.shape[0], len(data)), dtype="float32")
    # -inizializzo vettore tempo
    timestamp = np.zeros((len(data), 2), dtype="datetime64[D]")
    print("Importo i risultati di spostamento")
    tic = time.time()
    # -lancio loop
    for count, dat in enumerate(tqdm(data, desc="Loading displacement data")):
        # -leggo date
        # -t0 è la data slave, mentre t1 è la master
        t0 = np.datetime64(
            datetime.datetime.strptime(os.path.basename(dat)[8:16], "%Y%m%d")
        ).astype("datetime64[D]")
        t1 = np.datetime64(
            datetime.datetime.strptime(os.path.basename(dat)[17:25], "%Y%m%d")
        ).astype("datetime64[D]")
        # -popolo timestamp
        timestamp[count, 1] = t0
        timestamp[count, 0] = t1
        # -carico file
        file = np.loadtxt(dat, delimiter=",")
        EW[:, count] = file[:, 2]
        NS[:, count] = file[:, 3]
        WEIGHT[:, count] = file[:, 4]
    toc = time.time()
    print(f"Dati di spostamento importati in {(toc - tic) // 1} secondi")

    # -calcolo vettore deltaT
    deltat = np.abs(np.diff(timestamp, axis=1).astype("float32"))
    # ---creo vettore dei tempi dove ho un'osservazione
    # -1) cerco valori temporali per cui ho un dato
    Tu = np.sort(np.unique(timestamp))
    # -2) creo vettore degli intervalli su cui calcolare lo spostamento
    Time_hat = np.zeros([len(Tu) - 1, 2], dtype="datetime64[D]")
    for i in range(0, len(Tu) - 1):
        # print(i)
        Time_hat[i, :] = [Tu[i], Tu[i + 1]]
    # -calcolo deltat tra osservazioni
    DT_hat = np.diff(Time_hat, axis=1).astype("float").squeeze()
    # -inizializzo matrici dei risultati
    EW_hat = np.ones((meta.shape[0], len(DT_hat)), dtype="float32") * -9999
    NS_hat = np.ones((meta.shape[0], len(DT_hat)), dtype="float32") * -9999
    # -loop su tutta la mappa
    tic = time.time()
    print("Comincio time series inversion...")
    for ii in tqdm(range(EW.shape[0]), desc="Inverting time series"):
        # -estraggo spostamento e lo converto in metri
        ew = EW[ii, :].squeeze()
        ns = NS[ii, :].squeeze()
        weight = WEIGHT[ii, :].squeeze()
        # -se c'è anche un solo nan nella serie metto a nan il pixel e passo alla successiva iterata
        nanflag = np.any(np.isnan(ew)) or np.any(np.isnan(ns))
        ninenineflag = np.any(ew == -9999) or np.any(ns == -9999)
        if nanflag or ninenineflag:
            EW_hat[ii, :] = -9999
            NS_hat[ii, :] = -9999
            continue
        # ---creo vettore dei tempi dove ho un'osservazione
        # --- questo è inutile visto che l'ho già calcolato sopra ---
        # -1) cerco valori temporali per cui ho un dato
        Tu = np.sort(np.unique(timestamp))
        # -2) creo vettore degli intervalli su cui calcolare lo spostamento
        Time_hat = np.zeros([len(Tu) - 1, 2], dtype="datetime64[D]")
        for i in range(len(Tu) - 1):
            Time_hat[i, :] = [Tu[i], Tu[i + 1]]
        # -calcolo deltat tra osservazioni
        DT_hat = np.diff(Time_hat, axis=1).astype("float32").squeeze()
        # -3) inizializzo matrice di regressione M
        A = np.zeros((len(ew), Time_hat.shape[0])).astype("float32")
        # -4) popolo M
        for i in range(timestamp.shape[0]):
            pun = (Time_hat[:, 0] >= timestamp[i, 0]) & (
                Time_hat[:, 1] <= timestamp[i, 1]
            )
            A[i, pun] = 1
        # -observations
        n = A.shape[0]
        # -numero incognite
        p = A.shape[1]
        # -calcolo le due componenti
        comps = [ew, ns]
        for [enum, comp] in enumerate(comps):
            # -definisco i pesi
            # W0 = weight_definition(ew, ns, method="residuals", deltat=deltat)
            # W0 = weight_definition (ew, ns, method="Charrier", deltat=deltat,
            # EW=EW, NS=NS, c=(ii, jj), cc=None)
            # W0 = weight_definition(ew, ns, method="uniform")
            W0 = weight_definition(ew, ns, method="variable", cc=weight)

            # -regularisation term
            L = np.diag(np.ones(p) / DT_hat.squeeze()).astype("float32")
            for i in range(L.shape[0] - 1):
                L[i, i + 1] = -1 / DT_hat[i + 1]
            # -scaling parameter
            # regularization = np.std(comp)
            # regularization = 3*np.median(np.abs(comp-np.median(comp)))
            # l = np.mean(deltat) * regularization
            l = 1
            # print("Scaling term lambda:", l)
            # calcolo prima iterata
            X = GLS_solver(comp, A, weight=W0, regularisation=L, l=l)
            # -inizializzo fattore di improvement
            eps = 1
            # -inizializzo contatore
            count = 0
            # -se il miglioramento medio è minore di 0.01 o se supero le 10 iterate
            # -allora fermo la time series inversion
            while (eps > 0.01) and (count < iterates):
                # -aggiorno pesi
                if count == 0:
                    W = np.copy(W0)
                Wu, Ru = weight_update(
                    X, comp, A, weight=W, regularisation=L, W0=W0, l=l
                )
                # -se ci sono dei pesi a nan li pongo uguali a zero
                Wu[np.isnan(Wu)] = 0
                # -ricalcolo risultato
                Xu = GLS_solver(comp, A, weight=Wu, regularisation=L, l=l)
                # -calcolo improvement
                eps = np.mean(abs(X - Xu))
                X = np.copy(Xu)
                W = np.copy(Wu)
                count += 1
                # V_hat = X/DT_hat.squeeze()*365
                V_hat = X / DT_hat.squeeze()

            # -popolo output
            if enum == 0:
                EW_hat[ii, :] = V_hat
            elif enum == 1:
                NS_hat[ii, :] = V_hat

        # print(f"{ii} of {EW.shape[0]}")

    toc = time.time()
    print(f"Time series inversion ends in {(toc - tic) // 1} seconds")

    # -calcolo anche il tempo medio
    tm = timestamp[:, 0][:, None] + (deltat / 2 * 24).astype("timedelta64[h]")
    timestamp = np.hstack((timestamp, tm))

    TM = Time_hat[:, 0][:, None] + (DT_hat[:, None] / 2 * 24).astype("timedelta64[h]")
    Time_hat = np.hstack((Time_hat, TM))

    output = {
        "EW_hat": EW_hat,
        "NS_hat": NS_hat,
        "EW": EW,
        "NS": NS,
        "timestamp": timestamp,
        "deltat": deltat.squeeze(),
        "DT_hat": DT_hat.squeeze(),
        "Time_hat": Time_hat,
        "coordinates": meta[:, :2] * np.array([1, 1]),
    }

    return output


# -Fine


def fplot(O, pts):
    import matplotlib.pyplot as plt

    plt.ioff()
    """
    funzione per plottare le time series di una lista di punti
    """

    for ij in pts:
        i, j = ij[0], ij[1]
        idx = np.argmin(
            np.hypot(O["coordinates"][:, 0] - j, O["coordinates"][:, 1] - i)
        )

        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(
            O["timestamp"][:, 2],
            O["NS"][idx, :] / O["deltat"],
            marker="o",
            linestyle=" ",
        )
        plt.plot(O["Time_hat"][:, 2], O["NS_hat"][idx, :], marker="o", linestyle="")
        plt.legend(("Original", "Inversion"))
        plt.title(f"NS {O['coordinates'][idx, :]}")
        plt.subplot(2, 1, 2)
        plt.plot(
            O["timestamp"][:, 2],
            O["EW"][idx, :] / O["deltat"],
            marker="o",
            linestyle=" ",
        )
        plt.plot(O["Time_hat"][:, 2], O["EW_hat"][idx, :], marker="o", linestyle="")
        plt.legend(("Original", "Inversion"))
        plt.title(f"EW {O['coordinates'][idx, :]}")
        plt.show()


def save_inversion_results(
    results: dict,
    destination: str,
    base_name: str = "time_series_inversion",
    delimiter: str = ",",
):
    out_dir = Path(destination)
    out_dir.mkdir(parents=True, exist_ok=True)

    # save result object as pickle
    pickle.DEFAULT_PROTOCOL = pickle.HIGHEST_PROTOCOL
    with open(out_dir / f"{base_name}_results.pkl", "wb") as f:
        pickle.dump(results, f)

    # -salvo risultati
    header = [f"{x}" for x in results["Time_hat"][:, 2]]
    header.insert(0, "Y")
    header.insert(0, "X")
    header = f"{delimiter}".join(header)
    X = np.hstack((results["coordinates"], results["EW_hat"]))
    np.savetxt(
        out_dir / f"{base_name}_ew.txt", X, header=f"{header}", delimiter=delimiter
    )
    Y = np.hstack((results["coordinates"], results["NS_hat"]))
    np.savetxt(
        out_dir / f"{base_name}_ns.txt", Y, header=f"{header}", delimiter=delimiter
    )


if __name__ == "__main__":
    # -path alle mappe di spostamento
    source = "./data/"
    # -path dove salvare le time series
    destination = "./results/"

    output = run(
        source,
        iterates=10,
    )

    # -salvo risultati
    save_inversion_results(output, destination, base_name="ppcx_aug2021")

    #
    # -punti da plottare
    # -01_Jul2016-Oct2016
    # ij = [
    #     [1660, 2600],  # - centro ghiacciaio
    #     [2170, 2550],  # - fronte
    # ]
    # # -00
    # # ij = [ [1900, 1440],
    # #     ]

    # # -punto a cas
    # ij = [[2560, -1088]]
    # fplot(O, ij)
