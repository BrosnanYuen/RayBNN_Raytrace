

use arrayfire;


use rayon::prelude::*;


use std::collections::HashMap;


use rand::prelude::*;
use rand::distributions::{Distribution, Uniform};

use RayBNN_Sparse::Util::Convert::get_global_weight_idx;

use RayBNN_Sparse::Util::Search::find_unique;

use crate::Generate::Random::generate_random_uniform_rays;

use crate::Intersect::Sphere::line_sphere_intersect_batch;



const TWO_F64: f64 = 2.0;
const ONE_F64: f64 = 1.0;
const ZERO_F64: f64 = 0.0;

const EPSILON_F64: f64 = 1.0e-3;

const ONEMINUSEPSILON_F64: f64 = ONE_F64 - EPSILON_F64;

const RAYTRACE_LIMIT: u64 = 100000000;





/*
Raytracing algorithm 1 for creating neural connections. Randomly generates rays of random directions with variable number of random rays

Inputs
ray_num:        Number of rays per neuron per iteration
con_num:        Target number of total connections
netdata:        Network metadata
neuron_pos:     Neuron positions
neuron_idx:     Indexes of neuron positions

Outputs:
WRowIdxCOO:     Row vector in the COO sparse matrix
WColIdx:        Column vector in the COO sparse matrix

*/

pub fn RT1_random_rays<Z: arrayfire::RealFloating  >(
    modeldata_float: &HashMap<String, f64>,
    modeldata_int: &HashMap<String, u64>,


	neuron_pos: &arrayfire::Array<Z>,
	neuron_idx: &arrayfire::Array<i32>,



	WRowIdxCOO: &mut arrayfire::Array<i32>,
	WColIdx: &mut arrayfire::Array<i32>
)
{
    let neuron_size: u64 = modeldata_int["neuron_size"].clone();
    let input_size: u64 = modeldata_int["input_size"].clone();
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









	let neuron_dims = neuron_pos.dims();
	let neuron_num = neuron_dims[0];

	let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);








    let colseq = arrayfire::Seq::new(0.0, (space_dims-1) as f64, 1.0);

    let mut temparr = arrayfire::constant::<f64>(0.0,arrayfire::Dim4::new(&[neuron_size,space_dims,1,1]));

    let idx = neuron_idx.clone();

    let mut idxrs = arrayfire::Indexer::default();
    idxrs.set_index(&idx, 0, None);
	idxrs.set_index(&colseq, 1, Some(false));
    arrayfire::assign_gen(&mut temparr, &idxrs, neuron_pos);










	let active_size = neuron_idx.dims()[0];
	let active_size_u32 = active_size as u32;

	let circle_radius = arrayfire::constant::<f64>(neuron_rad,arrayfire::Dim4::new(&[active_size,1,1,1]));


    let mut curidxsel = arrayfire::rows(neuron_idx, (active_size-output_size) as i64, (active_size-1)   as i64);
	let mut idxsel = curidxsel.clone();

	let mut cur_num = idxsel.dims()[0];
	let mut cur_num_u32 = idxsel.dims()[0] as u32;




	let mut idxrs = arrayfire::Indexer::default();
    idxrs.set_index(&idxsel, 0, None);
	idxrs.set_index(&colseq, 1, Some(false));
    let mut cur_neuron_pos = arrayfire::index_gen(&temparr, idxrs);





	let mut start_line = arrayfire::constant::<f64>(0.0,single_dims);
    let mut dir_line = arrayfire::constant::<f64>(0.0,single_dims);



    generate_random_uniform_rays(
            &cur_neuron_pos,
            ray_num,
            con_rad,
        
            &mut start_line,
            &mut dir_line
        );


	let randarr_dims = arrayfire::Dim4::new(&[dir_line.dims()[0],1,1,1]);
	let randarr = arrayfire::randu::<f64>(randarr_dims);
	let (_, mut randidx) = arrayfire::sort_index(&randarr, 0, false);

	let mut raytrace_batch_size = 1 + ((RAYTRACE_LIMIT/(2*neuron_pos.dims()[0])) as u64);
	
	if randidx.dims()[0] > raytrace_batch_size
	{
		randidx = arrayfire::rows(&randidx, 0, (raytrace_batch_size-1)  as i64);
	}

	let mut idxrs1 = arrayfire::Indexer::default();
	let seq1 = arrayfire::Seq::new(0.0, (space_dims-1) as f64, 1.0);
	idxrs1.set_index(&randidx, 0, None);
	idxrs1.set_index(&seq1, 1, Some(false));
	start_line = arrayfire::index_gen(&start_line, idxrs1);

	
	let mut idxrs1 = arrayfire::Indexer::default();
	let seq1 = arrayfire::Seq::new(0.0, (space_dims-1) as f64, 1.0);
	idxrs1.set_index(&randidx, 0, None);
	idxrs1.set_index(&seq1, 1, Some(false));
	dir_line = arrayfire::index_gen(&dir_line, idxrs1);

	

	let mut line_num = start_line.dims()[0] as u32;

	let mut raytrace_batch_size = 1 + ((RAYTRACE_LIMIT/(start_line.dims()[0])) as u64);

	let mut intersect = arrayfire::constant::<bool>(false,single_dims);

	line_sphere_intersect_batch(
		raytrace_batch_size,
		&start_line,
		&dir_line,
	
		neuron_pos,
		&circle_radius,
	
		&mut intersect
		);


	intersect = arrayfire::reorder_v2(&intersect, 2, 1, Some(vec![0]));
	
	let mut idx_intersect = arrayfire::locate(&intersect);

	let mut div_idx = arrayfire::div(&idx_intersect,&active_size_u32, false);

	let mut mod_idx = arrayfire::modulo(&idx_intersect,&active_size_u32, false);

	let (key,value) = arrayfire::min_by_key(
		&div_idx,
		&mod_idx, 
		0
	);




	*WRowIdxCOO = arrayfire::modulo(&key,&cur_num_u32, false).cast::<i32>();

	*WColIdx = value.cast::<i32>();
	
	let mut global_idx = get_global_weight_idx(
		neuron_size,
		WRowIdxCOO,
		WColIdx,
	);


	global_idx = arrayfire::set_unique(&global_idx, false);


	*WRowIdxCOO = arrayfire::div(&global_idx,&neuron_size, false).cast::<i32>();

	*WColIdx = arrayfire::modulo(&global_idx,&neuron_size, false).cast::<i32>();





	*WRowIdxCOO = arrayfire::lookup(&idxsel, WRowIdxCOO, 0);
	






	

	curidxsel = WColIdx.clone();

	curidxsel = find_unique(&curidxsel, neuron_size);

	let mut newWRowIdxCOO = arrayfire::constant::<i32>(0,single_dims);

	let mut newWColIdx = arrayfire::constant::<i32>(0,single_dims);






	let mut start_lines1 = arrayfire::constant::<f64>(0.0,single_dims);
	let mut end_lines1 = arrayfire::constant::<f64>(0.0,single_dims);

	let mut start_lines2 = arrayfire::constant::<f64>(0.0,single_dims);
	let mut end_lines2 = arrayfire::constant::<f64>(0.0,single_dims);


	loop
	{


		idxsel = curidxsel.clone();

		cur_num = idxsel.dims()[0];
		cur_num_u32 = idxsel.dims()[0] as u32;
	
	
	
	
		let mut idxrs = arrayfire::Indexer::default();
		idxrs.set_index(&idxsel, 0, None);
		idxrs.set_index(&colseq, 1, Some(false));
		cur_neuron_pos = arrayfire::index_gen(&temparr, idxrs);
	
	
	
	
	
		start_line = arrayfire::constant::<f64>(0.0,single_dims);
		dir_line = arrayfire::constant::<f64>(0.0,single_dims);
	

		generate_random_uniform_rays(
				&cur_neuron_pos,
				ray_num,
				con_rad,
			
				&mut start_line,
				&mut dir_line
			);

			
		let randarr_dims = arrayfire::Dim4::new(&[dir_line.dims()[0],1,1,1]);
		let randarr = arrayfire::randu::<f64>(randarr_dims);
		let (_, mut randidx) = arrayfire::sort_index(&randarr, 0, false);

		let mut raytrace_batch_size = 1 + ((RAYTRACE_LIMIT/(2*neuron_pos.dims()[0])) as u64);
		
		if randidx.dims()[0] > raytrace_batch_size
		{
			randidx = arrayfire::rows(&randidx, 0, (raytrace_batch_size-1)  as i64);
		}

		let mut idxrs1 = arrayfire::Indexer::default();
		let seq1 = arrayfire::Seq::new(0.0, (space_dims-1) as f64, 1.0);
		idxrs1.set_index(&randidx, 0, None);
		idxrs1.set_index(&seq1, 1, Some(false));
		start_line = arrayfire::index_gen(&start_line, idxrs1);

		
		let mut idxrs1 = arrayfire::Indexer::default();
		let seq1 = arrayfire::Seq::new(0.0, (space_dims-1) as f64, 1.0);
		idxrs1.set_index(&randidx, 0, None);
		idxrs1.set_index(&seq1, 1, Some(false));
		dir_line = arrayfire::index_gen(&dir_line, idxrs1);

	
		line_num = start_line.dims()[0] as u32;
	
		raytrace_batch_size = 1 + ((RAYTRACE_LIMIT/(start_line.dims()[0])) as u64);
	
		intersect = arrayfire::constant::<bool>(false,single_dims);
	
		line_sphere_intersect_batch(
			raytrace_batch_size,
			&start_line,
			&dir_line,
		
			neuron_pos,
			&circle_radius,
		
			&mut intersect
			);
	
	
		intersect = arrayfire::reorder_v2(&intersect, 2, 1, Some(vec![0]));
		
		idx_intersect = arrayfire::locate(&intersect);
	
		div_idx = arrayfire::div(&idx_intersect,&active_size_u32, false);
	
		mod_idx = arrayfire::modulo(&idx_intersect,&active_size_u32, false);
	
		let (key,value) = arrayfire::min_by_key(
			&div_idx,
			&mod_idx, 
			0
		);
	
	

		newWRowIdxCOO = arrayfire::modulo(&key,&cur_num_u32, false).cast::<i32>();

		newWColIdx = value.cast::<i32>();




		newWRowIdxCOO = arrayfire::lookup(&idxsel, &newWRowIdxCOO, 0);














		*WRowIdxCOO = arrayfire::join(0, &newWRowIdxCOO, WRowIdxCOO);
	
		*WColIdx = arrayfire::join(0, &newWColIdx, WColIdx);


		
		global_idx = get_global_weight_idx(
			neuron_size,
			WRowIdxCOO,
			WColIdx,
		);
	
	
		global_idx = arrayfire::set_unique(&global_idx, false);
	
	
		*WRowIdxCOO = arrayfire::div(&global_idx,&neuron_size, false).cast::<i32>();
	
		*WColIdx = arrayfire::modulo(&global_idx,&neuron_size, false).cast::<i32>();
	
	


		if WRowIdxCOO.dims()[0] > con_num
		{
			break;
		}


		
	
		curidxsel = WColIdx.clone();
	
		curidxsel = find_unique(&curidxsel, neuron_size);
	
	}





}













