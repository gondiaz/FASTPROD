import numpy  as np

# general IC imports
NN = -999999

# IRENE
from invisible_cities.reco.peak_functions import split_in_peaks
from invisible_cities.reco.peak_functions import select_peaks
from invisible_cities.types.ic_types      import minmax

def signals_selected_splits(s1_indices, s2_indices,
                            s1_stride , s2_stride ,
                            s1_tmin, s1_tmax, s1_lmin, s1_lmax,
                            s2_tmin, s2_tmax, s2_lmin, s2_lmax):

    indices_split   = split_in_peaks(s1_indices, s1_stride)
    s1_selected_splits = select_peaks  (indices_split, 
                                        minmax(min = s1_tmin, max = s1_tmax), 
                                        minmax(min = s1_lmin, max = s1_lmax))

    indices_split   = split_in_peaks(s2_indices, s2_stride)
    s2_selected_splits = select_peaks  (indices_split, 
                                        minmax(min = s2_tmin, max = s2_tmax), 
                                        minmax(min = s2_lmin, max = s2_lmax))
    
    return s1_selected_splits, s2_selected_splits


def _1s1_1s2(pmt_ccwfs, s2_selected_splits, s1_selected_splits,
             s1emin   , s1wmin):
    ######## 1S1 1S2 CUT #########
    # 1S1 cut
    if len(s1_selected_splits)==0:
        return None
    # 1S2 cut
    if len(s2_selected_splits)>1:
        return None
        
    # S1 energy and width cut
    s1es, s1ws = [], []
    for ss in s1_selected_splits:
        s1_pmt = np.sum( pmt_ccwfs[:, ss[0]: ss[-1]], axis=0)
        s1es.append( np.sum(s1_pmt)    )
        s1ws.append( (ss[-1]-ss[0])*25 )
    s1es, s1ws = np.array(s1es), np.array(s1ws)

    sel = (s1es>=s1emin) & (s1ws>=s1wmin)
    idxs = np.argwhere(sel).flatten()

    if len(idxs)==0:
        return None
    elif len(idxs)>1:
        return None
    else:
        idx = idxs[0]
        s1_pmt = np.sum( pmt_ccwfs[:, s1_selected_splits[idx][0]: s1_selected_splits[idx][-1]], axis=0)
        times  = np.arange(s1_selected_splits[idx][0], s1_selected_splits[idx][-1])*25

        S1_time = times[np.argmax(s1_pmt)]
        return S1_time

def create_penthesilea_hits(s2_pmts_penth, s2_sipms_penth,
                            sipm_xs      , sipm_ys       , sipm_ids,
                            times        , S1_time):
    	###### create penthesilea hits ########
	datasipm = load_db.DataSiPM("new", run)
	sipm_xs  = datasipm.X.values
	sipm_ys  = datasipm.Y.values
	n_sipms = len(sipm_ids)
	
	X, Y = sipm_xs[sipm_ids], sipm_ys[sipm_ids]
	T = (times - S1_time)/1000
	
	E_per_slice = np.sum( s2_pmts_penth, axis=0)
	hits = []
	for t, e, q in zip(T, E_per_slice, s2_sipms_penth.T):
		hits.append( (X, Y, np.full( n_sipms, t),
			      np.full(n_sipms, e), np.full( n_sipms, -1),
			      q                  , np.full( n_sipms, -1) ) )
		
	hits = np.array( hits )
	hits = np.swapaxes(hits, axis1=1, axis2=2)
	hits = np.concatenate( hits )
	
	H = np.array(np.zeros(np.shape(hits)[0]),
		    dtype=[("X", int)  , ("Y", int)  , ("Z", float),
			   ("E", float), ("Ec",float),
			   ("Q", float), ("Qc", float)])
	H["X"], H["Y"], H["Z"] = hits[:, 0], hits[:, 1], hits[:, 2]
	H["E"], H["Ec"] = hits[:, 3], -1                            #OJO, la energía del hit es la energia de la slice
	H["Q"], H["Qc"] = hits[:, 5], -1
	
	#remove 0 charge hits
	sel = ~(H["Q"]==0)
	hits = H[sel]
    	return hits


def esmeralda_charge_cut(hits, qth_esmer):
    #### Charge cut ####
    sel = (hits["Q"]>=qth_esmer)
    hits["Q"][~sel] = 0

    slides = np.unique( hits["Z"] )
    for slide in slides:
        sel = (hits["Z"]==slide)
        slide_hits = hits[sel]
        q = slide_hits["Q"]
        e = slide_hits["E"]
        if np.sum( q ) == 0:
            idxs = np.argwhere(sel).flatten()
            hits = np.delete(hits, idxs)
            hits = np.insert(hits, 0, (0, 0, slide, np.sum(e), NN, -1))
        else:
            hits["E"][sel] = np.sum( e ) * q / np.sum(q)
    sel = (hits["Q"]==0)
    hits = np.delete( hits, np.argwhere(sel))
    hits = np.sort(hits, order="Z")
    return hits


def join_NN_hits(hits):
	# JOIN NN hits
	sel = (hits["Q"]==NN)
	
	nn_hits = hits[ sel]
	hits    = hits[~sel]
	
	slides = np.unique( hits["Z"] )
	for nn_hit in nn_hits:
		#select slide to append
		d = np.abs( slides - nn_hit["Z"] ) 
		slide = slides[ np.argmin( d ) ]
		slide_hits = hits[hits["Z"]==slide]
		
		#new energy
		new_E = np.sum(slide_hits["E"]) + nn_hit["E"]
		
		q = hits[hits["Z"]==slide]["Q"]
		Q = np.sum( q )
		
		hits["E"][hits["Z"] == slide] = new_E * q / Q

	return hits
