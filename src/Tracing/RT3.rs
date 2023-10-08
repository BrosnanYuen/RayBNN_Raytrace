

use arrayfire;


use rayon::prelude::*;


use std::collections::HashMap;


use rand::prelude::*;
use rand::distributions::{Distribution, Uniform};

use RayBNN_Sparse::Util::Convert::get_global_weight_idx;


const TWO_F64: f64 = 2.0;
const ONE_F64: f64 = 1.0;
const ZERO_F64: f64 = 0.0;

const EPSILON_F64: f64 = 1.0e-3;

const ONEMINUSEPSILON_F64: f64 = ONE_F64 - EPSILON_F64;



/*
Raytracing algorithm 3 for creating neural connections. Connects all neurons within minibatches/groups of neurons

Inputs
raytrace_options:    Raytracing options
netdata:             Network metadata
glia_pos_total:      The positions of all glial cells
input_pos_total:     Selected neurons positions as source for the rays
input_idx_total:     Selected neurons positions as source for the rays
hidden_pos_total:    Selected neurons positions as targets for the rays
hidden_idx_total:    Selected neurons positions as targets for the rays


Outputs:
WRowIdxCOO:     Row vector in the COO sparse matrix
WColIdx:        Column vector in the COO sparse matrix

*/

pub fn RT3_distance_limited_directly_connected<Z: arrayfire::RealFloating  >(
	raytrace_option_int: &HashMap<String, u64>,


    modeldata_float: &HashMap<String, f64>,
    modeldata_int: &HashMap<String, u64>,

	glia_pos_total: &arrayfire::Array<Z>,

	input_pos_total: &arrayfire::Array<Z>,
	input_idx_total: &arrayfire::Array<i32>,

	hidden_pos_total: &arrayfire::Array<Z>,
	hidden_idx_total: &arrayfire::Array<i32>,

	
	WRowIdxCOO: &mut arrayfire::Array<i32>,
	WColIdx: &mut arrayfire::Array<i32>
)
{

	let neuron_size: u64 = modeldata_int["neuron_size"].clone();
    let input_size: u64 = input_pos_total.dims()[0];
	let output_size: u64 = modeldata_int["output_size"].clone();
	let proc_num: u64 = modeldata_int["proc_num"].clone();
	let active_size: u64 = modeldata_int["active_size"].clone();
	let space_dims: u64 = modeldata_int["space_dims"].clone();
	let step_num: u64 = modeldata_int["step_num"].clone();







	let time_step: f64 = modeldata_float["time_step"].clone();
	let nratio: f64 = modeldata_float["nratio"].clone();
	let neuron_std: f64 = modeldata_float["neuron_std"].clone();
	let sphere_rad: f64 = modeldata_float["sphere_rad"].clone();
	let neuron_rad: f64 = modeldata_float["neuron_rad"].clone();
	let con_rad: f64 = modeldata_float["con_rad"].clone();
	let init_prob: f64 = modeldata_float["init_prob"].clone();




	let input_connection_num: u64 = raytrace_option_int["input_connection_num"].clone();
	let max_rounds: u64 = raytrace_option_int["max_rounds"].clone();
	let ray_glia_intersect: bool = raytrace_option_int["ray_glia_intersect"].clone() == 1;
	let ray_neuron_intersect: bool = raytrace_option_int["ray_neuron_intersect"].clone() == 1;


	let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);
	let TWO = arrayfire::constant::<f64>(TWO_F64,single_dims).cast::<Z>();

	let con_rad_Z = arrayfire::constant::<f64>(con_rad,single_dims).cast::<Z>();




}















