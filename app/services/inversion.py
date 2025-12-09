"""
Extraction of Velocity Time Series With an Optimal Temporal Sampling From
    Displacement Observation Networks - Charrier et al, TGRS, 2022

Fusion of multi-temporal and multi-sensor ice velocity observations -
    Charrier et al, ISPRS-ANNALS, 2022
"""

import datetime
import glob
import logging
import os
import time
from pathlib import Path
from typing import Literal

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

###=== Front-end functions ===###


def run_ts_inversion(
    day_dic_dir: Path,
    regularization: int = 10,
    iterates: int = 10,
):
    """
    Run time series inversion for all nodes in the dataset.

    Workflow:
    1) Load all DIC data with load_dic_data()
    2) Extract node information
    3) Call invert_node() for each node

    Args:
        day_dic_dir: Path to directory containing day_*.txt files
        regularization: Regularization parameter (for future use)
        iterates: Maximum number of iterations for inversion

    Returns:
        Dictionary with inversion results for all nodes
    """
    logger.info("Starting time series inversion for all nodes")

    # -Step 1: Load all data
    dic_data = load_dic_data(day_dic_dir)

    ew_data = dic_data["ew"]  # (n_observations, n_nodes)
    ns_data = dic_data["ns"]  # (n_observations, n_nodes)
    weight_data = dic_data["weight"]  # (n_observations, n_nodes)
    coordinates = dic_data["coordinates"]  # (n_nodes, 2)
    timestamp = dic_data["timestamp"]  # (n_observations, 2)
    deltat = dic_data["deltat"]  # (n_observations,)

    # -Step 2: Initialize output matrices
    n_nodes = ew_data.shape[1]
    n_times = len(np.unique(np.sort(timestamp))) - 1

    # --calcolo vettore deltaT
    # ---creo vettore dei tempi dove ho un'osservazione
    # -1) cerco valori temporali per cui ho un dato
    Tu = np.sort(np.unique(timestamp))
    Time_hat = np.zeros([len(Tu) - 1, 2], dtype="datetime64[D]")
    for i in range(len(Tu) - 1):
        Time_hat[i, :] = [Tu[i], Tu[i + 1]]
    DT_hat = np.diff(Time_hat, axis=1).astype("float").squeeze()

    # -inizializzo matrici dei risultati
    EW_hat = np.full((n_nodes, n_times), np.nan, dtype="float32")
    NS_hat = np.full((n_nodes, n_times), np.nan, dtype="float32")

    # -Step 3: Invert each node
    logger.info(f"Starting inversion for {n_nodes} nodes...")
    tic = time.time()
    for node_idx in tqdm(range(n_nodes), desc="Inverting nodes"):
        node_x = coordinates[node_idx, 0]
        node_y = coordinates[node_idx, 1]

        try:
            # -extract data for this node
            ew_node = ew_data[:, node_idx]
            ns_node = ns_data[:, node_idx]
            weight_node = weight_data[:, node_idx]

            # -invert the node
            result = invert_node(
                ew_series=ew_node,
                ns_series=ns_node,
                weight_series=weight_node,
                timestamp=timestamp,
                node_idx=node_idx,
                node_x=node_x,
                node_y=node_y,
                iterates=iterates,
            )

            # -populate results
            if result is not None:
                EW_hat[node_idx, :] = result["EW_hat"]
                NS_hat[node_idx, :] = result["NS_hat"]

        except Exception as e:
            # -keep nan for this node
            logger.warning(
                f"Error inverting node {node_idx} at ({node_x:.1f}, {node_y:.1f}): {e}"
            )

    toc = time.time()
    logger.info(f"Inversion completed in {(toc - tic) // 1} seconds")

    # -calcolo anche il tempo medio
    tm = timestamp[:, 0][:, None] + (deltat / 2 * 24).astype("timedelta64[h]")
    timestamp = np.hstack((timestamp, tm))
    TM = Time_hat[:, 0][:, None] + (DT_hat[:, None] / 2 * 24).astype("timedelta64[h]")
    Time_hat = np.hstack((Time_hat, TM))

    output = {
        "EW_hat": EW_hat,
        "NS_hat": NS_hat,
        "EW_init": ew_data,
        "NS_init": ns_data,
        "timestamp": timestamp,
        "deltat": deltat.squeeze(),
        "DT_hat": DT_hat.squeeze(),
        "Time_hat": Time_hat,
        "coordinates": coordinates,
    }

    return output


###=== INVERSION BACK-END FUNCTIONS ===####


def load_dic_data(day_dic_dir: Path):
    """
    Load all DIC displacement data from day_*.txt files.

    Args:
        day_dic_dir: Path to directory containing day_*.txt files

    Returns:
        Dictionary containing:
            - "ew": (n_observations, n_nodes) array of EW displacement
            - "ns": (n_observations, n_nodes) array of NS displacement
            - "weight": (n_observations, n_nodes) array of weights
            - "coordinates": (n_nodes, 2) array of node coordinates
            - "timestamp": (n_observations, 2) array of timestamps
            - "deltat": (n_observations,) array of time differences
    """
    logger.info("Loading DIC data from files...")

    day_dic_dir = Path(day_dic_dir)
    if not day_dic_dir.exists():
        raise FileNotFoundError(f"Directory {day_dic_dir} does not exist.")

    data_files = sorted(glob.glob(f"{day_dic_dir}/day*.txt"))
    if not data_files:
        raise FileNotFoundError(f"No day*.txt files found in {day_dic_dir}")

    # -leggo il primo file per determinare numero di nodi
    meta = np.loadtxt(data_files[0], delimiter=",")
    n_nodes = meta.shape[0]
    n_observations = len(data_files)

    # -inizializzo matrici
    ew_data = np.zeros((n_observations, n_nodes), dtype="float32")
    ns_data = np.zeros((n_observations, n_nodes), dtype="float32")
    weight_data = np.zeros((n_observations, n_nodes), dtype="float32")
    timestamp = np.zeros((n_observations, 2), dtype="datetime64[D]")
    coordinates = meta[:, :2]

    logger.info(f"Loading {n_observations} observations for {n_nodes} nodes...")
    tic = time.time()

    # -loop su tutti i file
    for count, dat_file in enumerate(tqdm(data_files, desc="Loading DIC data")):
        # -estraggo date dal nome del file
        t0 = np.datetime64(
            datetime.datetime.strptime(os.path.basename(dat_file)[8:16], "%Y%m%d")
        ).astype("datetime64[D]")
        t1 = np.datetime64(
            datetime.datetime.strptime(os.path.basename(dat_file)[17:25], "%Y%m%d")
        ).astype("datetime64[D]")
        timestamp[count, 1] = t0  # slave
        timestamp[count, 0] = t1  # master

        # -carico file
        file_data = np.loadtxt(dat_file, delimiter=",")

        # -estraggo i dati: colonne sono [x, y, dx, dy, weight]
        ew_data[count, :] = file_data[:, 2]
        ns_data[count, :] = file_data[:, 3]
        weight_data[count, :] = file_data[:, 4]

    toc = time.time()
    logger.info(f"Data loaded in {(toc - tic) // 1} seconds")

    # -calcolo vettore deltaT
    deltat = np.abs(np.diff(timestamp, axis=1).astype("float32")).squeeze()

    return {
        "ew": ew_data,
        "ns": ns_data,
        "weight": weight_data,
        "coordinates": coordinates,
        "timestamp": timestamp,
        "deltat": deltat,
    }


def GLS_solver(y, A, weight=None, regularisation=None, l=1.0):
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


def invert_node(
    ew_series: np.ndarray,
    ns_series: np.ndarray,
    weight_series: np.ndarray,
    timestamp: np.ndarray,
    node_idx: int,
    node_x: float,
    node_y: float,
    iterates: int = 10,
):
    """
    Run time series inversion for a single node.

    Args:
        ew_series: (n_observations,) array of EW displacement for this node
        ns_series: (n_observations,) array of NS displacement for this node
        weight_series: (n_observations,) array of weights for this node
        timestamp: (n_observations, 2) array of timestamps
        node_idx: Index of this node
        node_x: X coordinate of the node
        node_y: Y coordinate of the node
        iterates: Maximum number of iterations

    Returns:
        Dictionary with inversion results or None if data is invalid
    """

    # -estraggo spostamento
    ew = ew_series.squeeze()
    ns = ns_series.squeeze()
    weight = weight_series.squeeze()

    # -check per nan o -9999
    nanflag = np.any(np.isnan(ew)) or np.any(np.isnan(ns))
    ninenineflag = np.any(ew == -9999) or np.any(ns == -9999)

    if nanflag or ninenineflag:
        logger.debug(
            f"Node {node_idx} at ({node_x:.1f}, {node_y:.1f}) contains invalid data"
        )
        return None

    # -calcolo vettore deltaT
    deltat = np.abs(np.diff(timestamp, axis=1).astype("float32")).squeeze()

    # -creo vettore dei tempi per le osservazioni
    Tu = np.sort(np.unique(timestamp))
    Time_hat = np.zeros([len(Tu) - 1, 2], dtype="datetime64[D]")
    for i in range(len(Tu) - 1):
        Time_hat[i, :] = [Tu[i], Tu[i + 1]]
    DT_hat = np.diff(Time_hat, axis=1).astype("float").squeeze()

    # -inizializzo matrice di regressione A
    A = np.zeros((len(ew), Time_hat.shape[0])).astype("float32")
    for i in range(timestamp.shape[0]):
        pun = (Time_hat[:, 0] >= timestamp[i, 0]) & (Time_hat[:, 1] <= timestamp[i, 1])
        A[i, pun] = 1

    # -observations e incognite
    n = A.shape[0]
    p = A.shape[1]

    # -calcolo le due componenti
    EW_hat = np.zeros(len(DT_hat), dtype="float32")
    NS_hat = np.zeros(len(DT_hat), dtype="float32")

    comps = [ew, ns]
    comp_names = ["EW", "NS"]

    for enum, (comp, comp_name) in enumerate(zip(comps, comp_names)):
        # -definisco i pesi
        W0 = weight_definition(ew, ns, method="residuals", deltat=deltat)
        # W0 = weight_definition(
        #     ew, ns, method="Charrier", deltat=deltat, EW=EW, NS=NS, c=(ii, jj), cc=None
        # )
        # W0 = weight_definition(ew, ns, method="uniform")
        W0 = weight_definition(ew, ns, method="variable", cc=weight)

        # -regularisation term
        L = np.diag(np.ones(p) / DT_hat.squeeze()).astype("float32")
        for i in range(L.shape[0] - 1):
            L[i, i + 1] = -1 / DT_hat[i + 1]

        # -scaling parameter
        l = 1
        # regularization = np.std(comp)
        # regularization = 3*np.median(np.abs(comp-np.median(comp)))
        # l = np.mean(deltat) * regularization
        # print("Scaling term lambda:", l)

        # -calcolo prima iterata
        X = GLS_solver(comp, A, weight=W0, regularisation=L, l=l)
        V_hat = X / DT_hat.squeeze()

        # -inizializzo fattore di improvement eps e contatore
        eps = 1
        count = 0

        # -loop di iterazione
        # -se il miglioramento medio è minore di 0.01 o se supero le 10 iterate allora fermo la time series inversion
        while (eps > 0.01) and (count < iterates):
            # -aggiorno pesi
            if count == 0:
                W = np.copy(W0)
            Wu, Ru = weight_update(X, comp, A, weight=W, regularisation=L, W0=W0, l=l)
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
            EW_hat = V_hat
        elif enum == 1:
            NS_hat = V_hat

    # -calcolo anche il tempo medio
    tm = timestamp[:, 0][:, None] + (deltat[:, None] / 2 * 24).astype("timedelta64[h]")
    timestamp = np.hstack((timestamp, tm))
    TM = Time_hat[:, 0][:, None] + (DT_hat[:, None] / 2 * 24).astype("timedelta64[h]")
    Time_hat = np.hstack((Time_hat, TM))

    output = {
        "EW_hat": EW_hat,
        "NS_hat": NS_hat,
        "EW": ew,
        "NS": ns,
        "timestamp": timestamp,
        "deltat": deltat,
        "DT_hat": DT_hat.squeeze(),
        "Time_hat": Time_hat,
    }

    return output


def invert_node2(
    ew_series: np.ndarray,
    ns_series: np.ndarray,
    timestamp: np.ndarray,
    node_idx: int,
    node_x: float,
    node_y: float,
    weight_method: Literal[
        "residuals", "uniform", "Charrier", "variable"
    ] = "residuals",
    weight_variable: np.ndarray | None = None,
    regularization_method: Literal["laplacian", "none"] = "laplacian",
    lambda_scaling: float | str = 1.0,
    iterates: int = 10,
):
    """
    Run time series inversion for a single node.

    Args:
        ew_series: (n_observations,) array of EW displacement for this node
        ns_series: (n_observations,) array of NS displacement for this node
        weight_series: (n_observations,) array of weights for this node
        timestamp: (n_observations, 2) array of timestamps
        node_idx: Index of this node
        node_x: X coordinate of the node
        node_y: Y coordinate of the node
        iterates: Maximum number of iterations
        weight_method: Method for initial weight definition. Options:
            - "residuals": Based on MAD of residuals (default)
            - "uniform": Uniform weights
            - "variable": Use provided weight_variable array
            - "charrier": Spatial coherence method (requires EW/NS volumes)
        weight_variable: (n_observations,) array to use as initial weights
            when weight_method="variable". E.g., ensemble MAD for each timestamp.
        regularization_method: Method for regularization term. Options:
            - "laplacian": First-order Laplacian smoothing (default)
            - "none": No regularization
        lambda_scaling: Scaling parameter for regularization. Options:
            - float: Fixed value (default: 1.0)
            - "auto_std": lambda = mean(deltat) * std(component)
            - "auto_mad": lambda = mean(deltat) * 3*MAD(component)

    Returns:
        Dictionary with inversion results or None if data is invalid
    """

    # -estraggo spostamento
    ew = ew_series.squeeze()
    ns = ns_series.squeeze()

    # -check per nan o -9999
    nanflag = np.any(np.isnan(ew)) or np.any(np.isnan(ns))
    ninenineflag = np.any(ew == -9999) or np.any(ns == -9999)

    if nanflag or ninenineflag:
        logger.debug(
            f"Node {node_idx} at ({node_x:.1f}, {node_y:.1f}) contains invalid data"
        )
        return None

    # -calcolo vettore deltaT
    deltat = np.abs(np.diff(timestamp, axis=1).astype("float32")).squeeze()

    # -creo vettore dei tempi per le osservazioni
    Tu = np.sort(np.unique(timestamp))
    Time_hat = np.zeros([len(Tu) - 1, 2], dtype="datetime64[D]")
    for i in range(len(Tu) - 1):
        Time_hat[i, :] = [Tu[i], Tu[i + 1]]
    DT_hat = np.diff(Time_hat, axis=1).astype("float").squeeze()

    # -inizializzo matrice di regressione A
    A = np.zeros((len(ew), Time_hat.shape[0])).astype("float32")
    for i in range(timestamp.shape[0]):
        pun = (Time_hat[:, 0] >= timestamp[i, 0]) & (Time_hat[:, 1] <= timestamp[i, 1])
        A[i, pun] = 1

    # -observations e incognite
    n = A.shape[0]
    p = A.shape[1]

    # -calcolo le due componenti
    EW_hat = np.zeros(len(DT_hat), dtype="float32")
    NS_hat = np.zeros(len(DT_hat), dtype="float32")

    comps = [ew, ns]
    comp_names = ["EW", "NS"]

    for enum, (comp, comp_name) in enumerate(zip(comps, comp_names)):
        # -definisco i pesi iniziali in base al metodo scelto
        if weight_method == "residuals":
            W0 = weight_definition(ew, ns, method="residuals", deltat=deltat)
        elif weight_method == "uniform":
            W0 = weight_definition(ew, ns, method="uniform")
        elif weight_method == "variable":
            if weight_variable is None:
                raise ValueError(
                    "weight_variable must be provided when weight_method='variable'"
                )
            if len(weight_variable) != len(ew):
                raise ValueError(
                    f"weight_variable length {len(weight_variable)} must match "
                    f"ew_series length {len(ew)}"
                )
            W0 = weight_definition(ew, ns, method="variable", cc=weight_variable)
        elif weight_method == "charrier":
            # Note: Charrier method requires EW/NS which are not available
            # in single-node inversion. This would need to be passed as additional args.
            raise NotImplementedError(
                "Charrier method requires EW/NS volumes - not available in single-node mode"
            )
        else:
            raise ValueError(
                f"Invalid weight_method: {weight_method}. "
                "Valid options: 'residuals', 'uniform', 'variable', 'charrier'"
            )

            # -definisco termine di regolarizzazione
        if regularization_method == "laplacian":
            # First-order Laplacian smoothing
            L = np.diag(np.ones(p) / DT_hat.squeeze()).astype("float32")
            for i in range(L.shape[0] - 1):
                L[i, i + 1] = -1 / DT_hat[i + 1]
        elif regularization_method == "none":
            L = None
        else:
            raise ValueError(
                f"Invalid regularization_method: {regularization_method}. "
                "Valid options: 'laplacian', 'none'"
            )

        # -calcolo scaling parameter lambda
        if isinstance(lambda_scaling, (int, float)):
            l = float(lambda_scaling)
        elif lambda_scaling == "auto_std":
            l = np.mean(deltat) * np.std(comp)
            logger.debug(f"Auto lambda (std): {l:.4f}")
        elif lambda_scaling == "auto_mad":
            l = np.mean(deltat) * 3 * np.median(np.abs(comp - np.median(comp)))
            logger.debug(f"Auto lambda (MAD): {l:.4f}")
        else:
            raise ValueError(
                f"Invalid lambda_scaling: {lambda_scaling}. "
                "Valid options: float, 'auto_std', 'auto_mad'"
            )

        # -calcolo prima iterata
        X = GLS_solver(comp, A, weight=W0, regularisation=L, l=l)
        V_hat = X / DT_hat.squeeze()

        # -inizializzo fattore di improvement eps e contatore
        eps = 1
        count = 0

        # -loop di iterazione
        # -se il miglioramento medio è minore di 0.01 o se supero le 10 iterate allora fermo la time series inversion
        while (eps > 0.01) and (count < iterates):
            # -aggiorno pesi
            if count == 0:
                W = np.copy(W0)
            Wu, Ru = weight_update(X, comp, A, weight=W, regularisation=L, W0=W0, l=l)
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
            EW_hat = V_hat
        elif enum == 1:
            NS_hat = V_hat

    # -calcolo anche il tempo medio
    tm = timestamp[:, 0][:, None] + (deltat[:, None] / 2 * 24).astype("timedelta64[h]")
    timestamp = np.hstack((timestamp, tm))
    TM = Time_hat[:, 0][:, None] + (DT_hat[:, None] / 2 * 24).astype("timedelta64[h]")
    Time_hat = np.hstack((Time_hat, TM))

    output = {
        "EW_hat": EW_hat,
        "NS_hat": NS_hat,
        "EW": ew,
        "NS": ns,
        "timestamp": timestamp,
        "deltat": deltat,
        "DT_hat": DT_hat.squeeze(),
        "Time_hat": Time_hat,
    }

    return output


if __name__ == "__main__":
    # -path alle mappe di spostamento
    cwd = Path(__file__).parents[1]
    day_dic_dir = cwd / "data/day_dic"

    output = run_ts_inversion(
        day_dic_dir=day_dic_dir,
        iterates=10,
    )

    # Plot output of a sample point to stdout for quick verification
    sample_idx = 100  # Change this index to plot different points
    print("Sample point time series:")
    print("East-West Velocity:", output["EW_hat"][sample_idx, :])
    print("North-South Velocity:", output["NS_hat"][sample_idx, :])
