import sys
import glob
import numpy  as np
import tables as tb

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# general IC imports
from invisible_cities.core.configure         import configure
from invisible_cities.database               import load_db
from invisible_cities.core.system_of_units_c import units
adc, pes, mus = units.adc, units.pes, units.mus
NN = -999999

# IRENE
from invisible_cities.cities.components import deconv_pmt
from invisible_cities.cities.components import calibrate_pmts
from invisible_cities.cities.components import calibrate_sipms

from invisible_cities.cities.components import deconv_pmt
from invisible_cities.cities.components import calibrate_pmts
from invisible_cities.cities.components import calibrate_sipms
from invisible_cities.cities.components import zero_suppress_wfs

from invisible_cities.reco.peak_functions import split_in_peaks
from invisible_cities.reco.peak_functions import select_peaks
from invisible_cities.reco.peak_functions import select_wfs_above_time_integrated_thr
from invisible_cities.reco.peak_functions import pick_slice_and_rebin

from invisible_cities.types.ic_types import minmax

# PENTHESILEA
from invisible_cities.reco.peak_functions import rebin_times_and_waveforms

# ESMERALDA
from invisible_cities.reco.corrections_new import read_maps
from invisible_cities.reco.corrections_new import apply_all_correction
from invisible_cities.reco.corrections_new import norm_strategy

# LT CORRECTIONS
from invisible_cities.reco.corrections_new import correct_lifetime_
from invisible_cities.reco.corrections_new import maps_coefficient_getter

# UTILS
import utils


def get_config_params(config):
	# cut params
	s1emin = config.s1emin
	s1wmin = config.s1wmin
	#nS1s = config.nS1s
	#nS2s = config.nS2s
	pmt_ids = np.array( config.active_pmts )
	
	#IRENE params
	run        = config.run
	files_in   = config.files_in
	nfin       = config.nfin
	file_out   = config.file_out
	n_baseline = config.n_baseline
	n_mau      = config.n_mau
	thr_mau    = config.thr_mau
	thr_csum_s1 = config.thr_csum_s1
	thr_csum_s2 = config.thr_csum_s2
	thr_sipm      = config.thr_sipm
	thr_sipm_type = config.thr_sipm_type
	s1_tmin         = config.s1_tmin
	s1_tmax         = config.s1_tmax
	s1_stride       = config.s1_stride
	s1_lmin         = config.s1_lmin
	s1_lmax         = config.s1_lmax
	s1_rebin_stride = config.s1_rebin_stride
	s2_tmin         = config.s2_tmin
	s2_tmax         = config.s2_tmax
	s2_stride       = config.s2_stride
	s2_lmin         = config.s2_lmin
	s2_lmax         = config.s2_lmax
	s2_rebin_stride = config.s2_rebin_stride
	thr_sipm_s2 = config.thr_csum_s2
	detector_db = config.detector_db
	
	#Penthesilea
	qth_penth = config.qth_penth
	rebin     = config.rebin
	
	#Esmeralda
	qth_esmer  = config.qth_esmer  #NOT USED
	map_file   = config.map_file
	apply_temp = config.apply_temp
	
	if thr_sipm_type.lower() == "common":
		sipm_thr = thr_sipm

	return s1emin, s1wmin, pmt_ids,\
	       run, files_in, nfin, file_out, n_baseline,\
	       n_mau, thr_mau, thr_csum_s1, thr_csum_s2, sipm_thr,\
	       s1_tmin, s1_tmax, s1_stride, s1_lmin, s1_lmax, s1_rebin_stride,\
	       s2_tmin, s2_tmax, s2_stride, s2_lmin, s2_lmax ,s2_rebin_stride,\
	       thr_sipm_s2, detector_db,\
	       qth_penth, rebin,\
	       qth_esmer,  map_file, apply_temp


def fast_prod(s1emin, s1wmin, pmt_ids,\
              run, files_in, nfin, file_out, n_baseline,\
              n_mau, thr_mau, thr_csum_s1, thr_csum_s2, sipm_thr,\
              s1_tmin, s1_tmax, s1_stride, s1_lmin, s1_lmax, s1_rebin_stride,\
              s2_tmin, s2_tmax, s2_stride, s2_lmin, s2_lmax ,s2_rebin_stride,\
              thr_sipm_s2, detector_db,\
              qth_penth, rebin,\
              qth_esmer,  map_file, apply_temp):

	# FILE OUT
	h5file = tb.open_file(file_out, mode="w", title="Fast Prod")

	group = h5file.create_group("/", "Summary", "Summary")
	h5file.create_earray(group, "Zmin", tb.Float64Atom(), shape=(0, ))
	h5file.create_earray(group, "Zmax", tb.Float64Atom(), shape=(0, ))
	h5file.create_earray(group, "DZ", tb.Float64Atom(), shape=(0, ))
	h5file.create_earray(group, "E" , tb.Float64Atom(), shape=(0, ))
	h5file.create_earray(group, "Q" , tb.Float64Atom(), shape=(0, ))
	h5file.create_earray(group, "Qc", tb.Float64Atom(), shape=(0, ))
	group  = h5file.create_group("/", f"Event_Info", "Info")
	class Event_Info(tb.IsDescription):
		event = tb.Int32Col ()
		time  = tb.UInt64Col()
	Event_Info_table = h5file.create_table(group, "Event_Time", Event_Info, "Event_Time")
	EI = Event_Info_table.row

	files_in = glob.glob( files_in )
	files_in.sort()
	files_in = files_in[:nfin]

	# DATASIPMs AND MAPS
	datasipm = load_db.DataSiPM("new", run)
	sipm_xs  = datasipm.X.values
	sipm_ys  = datasipm.Y.values
	
	qmaps = read_maps( qmap_file )
	get_lt_corr_fun = maps_coefficient_getter(qmaps.mapinfo, qmaps.lt)
	if apply_temp:
		raise Exception("Apply temp is False")
	else:
		ltevol_vs_t = lambda x : np.ones_like(x)
	
	# FAST PROD
	_Zmin, _Zmax, _DZ, _E, _Q, _Qc = [], [], [], [], []
	for file in files_in:

		print(file)

		RWFs_file = tb.open_file( file )
		pmt_rwfs_all  = RWFs_file.root.RD.pmtrwf
		sipm_rwfs_all = RWFs_file.root.RD.sipmrwf
		time_stamps   = RWFs_file.root.Run.events.read()
		
		for event_time, pmt_rwfs, sipm_rwfs in zip(time_stamps, pmt_rwfs_all, sipm_rwfs_all):
		
			################################
			############ IRENE #############
			################################
			
			#pmt processing
			rwf_to_cwf  = deconv_pmt    (detector_db, run, n_baseline)
			pmt_cwfs    = rwf_to_cwf    (pmt_rwfs)
			cwf_to_ccwf = calibrate_pmts(detector_db, run, n_mau, thr_mau)
			pmt_ccwfs, ccwfs_mau, cwf_sum, cwf_sum_mau  = cwf_to_ccwf(pmt_cwfs)
			
			#sipm processing
			sipm_rwf_to_cal = calibrate_sipms(detector_db, run, sipm_thr)
			sipm_cwfs = sipm_rwf_to_cal(sipm_rwfs)
			
			
			#Find signals
			zero_suppress = zero_suppress_wfs(thr_csum_s1, thr_csum_s2)
			s1_indices, s2_indices = zero_suppress(cwf_sum, cwf_sum_mau)
			
			s1_selected_splits,\
			s2_selected_splits = utils.signals_selected_splits(s1_indices, s2_indices,
									   s1_stride , s2_stride,
									   s1_tmin   , s1_tmax   , s1_lmin, s1_lmax,
									   s2_tmin   , s2_tmax   , s2_lmin, s2_lmax)
			
			######## 1S1 1S2 CUT ##########
			S1_time = utils._1s1_1s2(pmt_ccwfs, s2_selected_splits, s1_selected_splits,
						 s1emin   , s1wmin)
			if not S1_time: continue
			
			# Rebin S2_pmts
			times, rebinned_widths, s2_pmts = pick_slice_and_rebin(s2_selected_splits[0],
									       np.arange(pmt_ccwfs.shape[1]) * 25 * units.ns,
									       np.full  (pmt_ccwfs.shape[1],   25 * units.ns),
									       pmt_ccwfs,
									       rebin_stride = s2_rebin_stride,
									       pad_zeros    = True)
			#select and thr_sipm_s2
			s2_sipms = sipm_cwfs[:, s2_selected_splits[0][0] //40 : s2_selected_splits[0][-1]//40 + 1]
			sipm_ids, s2_sipms = select_wfs_above_time_integrated_thr(s2_sipms, thr_sipm_s2)
			
			######## IRENE FINAL S2 WFS #######
			s2_pmts  = np.float32( s2_pmts )
			s2_sipms = np.float32( s2_sipms)
			times    = np.float32( times   )
			#select pmt_ids wfs
			c = np.zeros(s2_pmts.shape[0])
			c[pmt_ids] = 1
			s2_pmts  = np.multiply( c, s2_pmts.T ).T

			################################
			######## PENTHESILEA ###########
			################################
			
			########## Rebin ############
			_,     _, s2_sipms = rebin_times_and_waveforms(times, rebinned_widths, s2_sipms,
                                        			       rebin_stride=rebin, slices=None)
			times, _, s2_pmts  = rebin_times_and_waveforms(times, rebinned_widths, s2_pmts,
                                        			       rebin_stride=rebin, slices=None)
			s2_pmts_penth  = np.copy( s2_pmts )
			s2_sipms_penth = np.copy( s2_sipms )
			
			###### create penthesilea hits ########
			hits = utils.create_penthesilea_hits(s2_pmts_penth, s2_sipms_penth,
							     sipm_xs      , sipm_ys       , sipm_ids,
							     times        , S1_time)

			######## correct penthesilea charge ########
			X, Y, Z = hits["X"], hits["Y"], hits["Z"]
			Q = hits["Q"]
			T = np.full(len(hits), event_time[-1]/1000)

			lt_factor  = correct_lifetime_(Z, get_lt_corr_fun(X, Y) * ltevol_vs_t(T))
			Qc = lt_factor * Q

			hits["Qc"] = Qc

			######### penthesilea charge cut ##########
			qth = qth_penth
			
			sel = (hits["Qc"]>=qth)
			hits["Qc"][~sel] = 0
			
			slides = np.unique( hits["Z"] )
			for slide in slides:
    				sel = (hits["Z"]==slide)
    				slide_hits = hits[sel]
    				
    				q = slide_hits["Qc"]
    				e = slide_hits["E"]
    				slide_e = e[0]     ## OJO AQUÍ
				
    				if np.sum( q ) == 0:
        				idxs = np.argwhere(sel).flatten()
        				hits = np.delete(hits, idxs)
        				hits = np.insert(hits, 0, (0, 0, slide,
                                   				slide_e, NN,
                                   				NN, NN))
    				else:
        				hits["E"][sel] = slide_e * q / np.sum(q)
        			
			sel = (hits["Qc"]==0)
			hits = np.delete( hits, np.argwhere(sel))
			hits = np.sort(hits, order="Z")

			###### join NN hits ###
			hits = utils.join_NN_hits( hits )


			###########################
			####### APPEND DATA #######
			###########################
			# Event Info
			EI["event"] = event_time[0]
			EI["time"]  = event_time[1]
			EI.append()
			
			## Z, DZ, E, Q, Ec
			Z, E, Q, Qc = hits["Z"], hits["E"], hits["Q"], hits["Qc"]
			Qc[ np.isnan(Qc) ] = 0
			
			_Zmax.append( np.max(Z) )
			_Zmin.append( np.min(Z) )
			_DZ.append( np.max(Z) - np.min(Z) )
			_E .append( np.sum(E)  )
			_Q .append( np.sum(Q)  )
			_Qc.append( np.sum(Qc) )
			
		# close RWF file
		RWFs_file.close()
	
	h5file.root.Summary.Z .append( _Z  )
	h5file.root.Summary.DZ.append( _DZ )
	h5file.root.Summary.E .append( _E  )
	h5file.root.Summary.Q .append( _Q  )
	h5file.root.Summary.Ec.append( _Ec )
	
	#write to disk
	h5file.flush()
	h5file.close()
	
	
if __name__ == "__main__":
	config = configure(sys.argv).as_namespace
	fast_prod(*get_config_params(config))
