

use arrayfire;


use rayon::prelude::*;


use std::collections::HashMap;


use rand::prelude::*;
use rand::distributions::{Distribution, Uniform};

use RayBNN_Sparse::Util::Convert::get_global_weight_idx;

use crate::Generate::Fixed::filter_rays;

use crate::Generate::Fixed::tileDown;

use crate::Intersect::Sphere::line_sphere_intersect_batchV2;

use crate::Generate::Fixed::rays_from_neuronsA_to_neuronsB;





const TWO_F64: f64 = 2.0;
const ONE_F64: f64 = 1.0;
const ZERO_F64: f64 = 0.0;

const EPSILON_F64: f64 = 1.0e-3;

const ONEMINUSEPSILON_F64: f64 = ONE_F64 - EPSILON_F64;

const RAYTRACE_LIMIT: u64 = 100000000;


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

pub fn RT3_distance_limited_directly_connected<Z: arrayfire::RealFloating<AggregateOutType = Z>  >(
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




	let input_connection_num: u64 = modeldata_int["ray_input_connection_num"].clone();
	let max_rounds: u64 = modeldata_int["ray_max_rounds"].clone();
	let ray_glia_intersect: bool = modeldata_int["ray_glia_intersect"].clone() == 1;
	let ray_neuron_intersect: bool = modeldata_int["ray_neuron_intersect"].clone() == 1;


	let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);

	let ZERO = arrayfire::constant::<f64>(ZERO_F64,single_dims).cast::<Z>();

	let neuron_rad_Z = arrayfire::constant::<f64>(neuron_rad,single_dims).cast::<Z>();








	let mut gidxOld = arrayfire::constant::<u64>(0,single_dims);

	let mut gidxOld_cpu:Vec<u64> = Vec::new();

	let WColIdxelements =  WColIdx.elements();

	//let mut WValues_cpu = Vec::new();
	let mut WRowIdxCOO_cpu = Vec::new();
	let mut WColIdx_cpu = Vec::new();

	//Insert previous COO Matrix values
	if WColIdxelements > 1
	{

		//TO CPU
		

		WRowIdxCOO_cpu = vec!(i32::default();WRowIdxCOO.elements());
		WRowIdxCOO.host(&mut WRowIdxCOO_cpu);

		WColIdx_cpu = vec!(i32::default();WColIdx.elements());
		WColIdx.host(&mut WColIdx_cpu);


		//Compute global index
		gidxOld = get_global_weight_idx(
			neuron_size,
			&WRowIdxCOO,
			&WColIdx,
		);

		//TO CPU
		gidxOld_cpu = vec!(u64::default();gidxOld.elements());
		gidxOld.host(&mut gidxOld_cpu);
		drop(gidxOld);


		
		*WRowIdxCOO = arrayfire::constant::<i32>(0,single_dims);
		*WColIdx = arrayfire::constant::<i32>(0,single_dims);

	}
















	let mut hidden_idx = arrayfire::constant::<i32>(0,single_dims);

 
	let mut tiled_input_idx = arrayfire::constant::<i32>(0,single_dims);

	let mut tiled_hidden_idx = arrayfire::constant::<i32>(0,single_dims);


	//Get input and hidden positions

	let mut hidden_pos =  ZERO.clone();

 

	//let mut circle_radius = arrayfire::constant::<f64>(neuron_rad,arrayfire::Dim4::new(&[hidden_pos.dims()[0],1,1,1])).cast::<Z>();

	let mut circle_radius = arrayfire::tile(&neuron_rad_Z, arrayfire::Dim4::new(&[hidden_pos.dims()[0],1,1,1]));


	let mut start_line = ZERO.clone();
    let mut dir_line = ZERO.clone();



	let mut raytrace_batch_size = 1 + ((RAYTRACE_LIMIT/start_line.dims()[0]) as u64);

	


	let mut hidden_size_u32 = hidden_pos_total.dims()[0] as u32;
	let mut hidden_size = hidden_pos_total.dims()[0];

	
	//Store COO Matrix values


	let mut join_all = Vec::new();



	let mut gidx1 = arrayfire::constant::<u64>(0,single_dims);

	let mut gidx1_cpu:Vec<u64> = Vec::new();


	let mut prev_con_num = 0;
	


	let mut input_idx_size = 0;

	let mut rng = rand::thread_rng();
	let rand_vec: Vec<u64> = (0..input_size).collect();
	let mut select_input_idx: u64 = 0;


	let mut input_pos = ZERO.clone();
	let mut input_idx  = arrayfire::constant::<i32>(0,single_dims);


	let mut glia_pos = ZERO.clone();
	let mut glia_idx  = arrayfire::constant::<i32>(0,single_dims);


	let mut same_counter: u64 = 0;
	
	let mut nonoverlapping = true;

	for vv in 0..max_rounds
	{
		select_input_idx = rand_vec.choose(&mut rng).unwrap().clone();
		let mut target_input = arrayfire::row(input_pos_total, select_input_idx as i64);
		

		


		input_pos = input_pos_total.clone();
		input_idx  = input_idx_total.clone();

		filter_rays(
			2.0f64*con_rad,
		
			&target_input,
		
			&mut input_pos,
			&mut input_idx,
		);

		if input_idx.dims()[0] == 0
		{
			continue;
		}

		input_idx_size = input_idx.dims()[0];


		hidden_pos = hidden_pos_total.clone();
		hidden_idx  = hidden_idx_total.clone();

		filter_rays(
			con_rad,
		
			&target_input,
		
			&mut hidden_pos,
			&mut hidden_idx,
		);

		hidden_size = hidden_idx.dims()[0];
		hidden_size_u32 = hidden_idx.dims()[0] as u32;

		if hidden_size == 0
		{
			continue;
		}

		


	


		//Generate rays starting from input neurons
		start_line = ZERO.clone();
		dir_line = ZERO.clone();

		

		let tile_dims = arrayfire::Dim4::new(&[hidden_size,1,1,1]);

		tiled_input_idx =  arrayfire::tile(&input_idx, tile_dims);
		drop(input_idx);
		
		tiled_hidden_idx = hidden_idx.clone();
		drop(hidden_idx);

		tileDown(
			input_idx_size,
		
			&mut tiled_hidden_idx
		);

		//println!("z1");
		//println!("input_pos.dims()[0] {}",input_pos.dims()[0]);
		//println!("input_pos.dims()[1] {}",input_pos.dims()[1]);
		//println!("hidden_pos.dims()[0] {}",hidden_pos.dims()[0]);
		//println!("hidden_pos.dims()[1] {}",hidden_pos.dims()[1]);
		//println!("con_rad {}", con_rad);

		rays_from_neuronsA_to_neuronsB(
			con_rad,

			&input_pos,
			&hidden_pos,
		
			&mut start_line,
			&mut dir_line,

			&mut tiled_input_idx,
			&mut tiled_hidden_idx,
		);
		drop(input_pos);

		//println!("z1");
		//println!("start_line.dims()[0] {}",start_line.dims()[0]);
		//println!("start_line.dims()[1] {}",start_line.dims()[1]);
		//println!("dir_line.dims()[0] {}",dir_line.dims()[0]);
		//println!("dir_line.dims()[1] {}",dir_line.dims()[1]);
		//println!("tiled_input_idx.dims()[0] {}",tiled_input_idx.dims()[0]);
		//println!("tiled_input_idx.dims()[1] {}",tiled_input_idx.dims()[1]);
		//println!("tiled_hidden_idx.dims()[0] {}",tiled_hidden_idx.dims()[0]);
		//println!("tiled_hidden_idx.dims()[1] {}",tiled_hidden_idx.dims()[1]);

		if start_line.dims()[0] == 0
		{
			continue;
		}


	
		raytrace_batch_size = 1 + ((RAYTRACE_LIMIT/start_line.dims()[0]) as u64);
	
		
		//circle_radius = arrayfire::constant::<f64>(neuron_rad,arrayfire::Dim4::new(&[hidden_pos.dims()[0],1,1,1])).cast::<Z>();

		circle_radius = arrayfire::tile(&neuron_rad_Z, arrayfire::Dim4::new(&[hidden_pos.dims()[0],1,1,1]));

		
		if ray_neuron_intersect && (hidden_size > 1)
		{
			line_sphere_intersect_batchV2(
				raytrace_batch_size,
			
				2,
			
				&hidden_pos,
				&circle_radius,
			
				&mut start_line,
				&mut dir_line,
			
				&mut tiled_input_idx,
				&mut tiled_hidden_idx,
			);
		}
		drop(hidden_pos);

		if tiled_input_idx.dims()[0] == 0
		{
			continue;
		}

		//println!("a1");
		//println!("intersect.dims()[0] {}",intersect.dims()[0]);
		//println!("intersect.dims()[1] {}",intersect.dims()[1]);
		//println!("intersect.dims()[2] {}",intersect.dims()[2]);

		
		//println!("a2");
		//println!("intersect.dims()[0] {}",intersect.dims()[0]);
		//println!("intersect.dims()[1] {}",intersect.dims()[1]);
		//println!("intersect.dims()[2] {}",intersect.dims()[2]);

		
		//println!("input_size {}",input_size);




		glia_pos = glia_pos_total.clone();
		glia_idx  = arrayfire::constant::<i32>(0,glia_pos.dims());

		filter_rays(
			con_rad,
		
			&target_input,
		
			&mut glia_pos,
			&mut glia_idx,
		);
		drop(glia_idx);

		raytrace_batch_size = 1 + ((RAYTRACE_LIMIT/start_line.dims()[0]) as u64);
	
		//circle_radius = arrayfire::constant::<f64>(neuron_rad,arrayfire::Dim4::new(&[glia_pos.dims()[0],1,1,1])).cast::<Z>();
		
		circle_radius = arrayfire::tile(&neuron_rad_Z, arrayfire::Dim4::new(&[glia_pos.dims()[0],1,1,1]));


		if ray_glia_intersect && (glia_pos.dims()[0] > 1)
		{
			line_sphere_intersect_batchV2(
				raytrace_batch_size,
			
				0,
			
				&glia_pos,
				&circle_radius,
			
				&mut start_line,
				&mut dir_line,
			
				&mut tiled_input_idx,
				&mut tiled_hidden_idx,
			);
		}
		drop(glia_pos);

		

		if tiled_input_idx.dims()[0] == 0
		{
			continue;
		}

		


		//Compute global index
		gidx1 = get_global_weight_idx(
			neuron_size,
			&tiled_hidden_idx,
			&tiled_input_idx,
		);

		let mut gidx1_cpu = vec!(u64::default();gidx1.elements());
		gidx1.host(&mut gidx1_cpu);
		drop(gidx1);

		let mut tiled_hidden_idx_cpu = vec!(i32::default();tiled_hidden_idx.elements());
		tiled_hidden_idx.host(&mut tiled_hidden_idx_cpu);
		drop(tiled_hidden_idx);

		let mut tiled_input_idx_cpu = vec!(i32::default();tiled_input_idx.elements());
		tiled_input_idx.host(&mut tiled_input_idx_cpu);
		drop(tiled_input_idx);

		//println!("join_WColIdx.len() {}", join_WColIdx.len());
		//println!("join_WColIdx.keys().len() {}", join_WColIdx.keys().len());
		//println!("gidx1_cpu.len() {}", gidx1_cpu.len());


		//Save new neural connections to COO matrix hashmap
		for qq in 0..gidx1_cpu.len()
		{
			join_all.push( (gidx1_cpu[qq].clone(),tiled_input_idx_cpu[qq].clone(),tiled_hidden_idx_cpu[qq].clone()) );
			
		}

		

		//println!("join_WColIdx.len() {}", join_WColIdx.len());
		if ((join_all.len() as u64) > (input_connection_num))
		{
			join_all.par_sort_unstable_by_key(|pair| pair.0);
			join_all.dedup_by_key(|pair| pair.0);

			if ((join_all.len() as u64) > (input_connection_num))
			{
				break;
			}
		}

		if ((join_all.len() as u64) > prev_con_num)
		{
			prev_con_num = join_all.len() as u64;
			same_counter = 0;
		}
		else
		{
			same_counter = same_counter + 1;
		}

		if same_counter > 5
		{
			break;
		}


	}


	drop(gidx1_cpu);
	//Insert previous COO Matrix values
	if WColIdxelements > 1
	{

		//Insert old value
		for qq in 0..gidxOld_cpu.len()
		{
			join_all.push( (gidxOld_cpu[qq].clone(),WColIdx_cpu[qq].clone(),WRowIdxCOO_cpu[qq].clone()) );

			
		}
		drop(gidxOld_cpu);
		//println!("join_WValues.len() {}", join_WValues.len());

	}


	//Sort global index
	join_all.par_sort_unstable_by_key(|pair| pair.0);
	join_all.dedup_by_key(|pair| pair.0);



	let (WColIdx_cpu, WRowIdxCOO_cpu): (Vec<_>, Vec<_>) = join_all.par_iter().cloned().map(|(_,b,c)| (b,c)).unzip();



	//Convert cpu vector to gpu array
	*WColIdx = arrayfire::Array::new(&WColIdx_cpu, arrayfire::Dim4::new(&[WColIdx_cpu.len() as u64, 1, 1, 1]));
	*WRowIdxCOO = arrayfire::Array::new(&WRowIdxCOO_cpu, arrayfire::Dim4::new(&[WColIdx_cpu.len() as u64, 1, 1, 1]));
	

}















