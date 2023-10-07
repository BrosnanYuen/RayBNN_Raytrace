use arrayfire;


use rayon::prelude::*;


use std::collections::HashMap;


const TWO_F64: f64 = 2.0;
const ONE_F64: f64 = 1.0;
const ZERO_F64: f64 = 0.0;





pub fn tileDown<Z: arrayfire::HasAfEnum >(
	repeat_num: u64,

	input_arr: &mut arrayfire::Array<Z>
	)
{
	let space_dims: u64 = input_arr.dims()[1];

	let input_arr_num: u64 = input_arr.dims()[0];

	let tile_dims = arrayfire::Dim4::new(&[1,repeat_num,1,1]);

	*input_arr = arrayfire::tile(input_arr, tile_dims);

	*input_arr = arrayfire::transpose(input_arr, false);

	let dims = arrayfire::Dim4::new(&[space_dims, repeat_num*input_arr_num , 1 , 1]);
	*input_arr = arrayfire::moddims(input_arr, dims);

	*input_arr = arrayfire::transpose(input_arr, false);


}







pub fn filter_rays<Z: arrayfire::RealFloating<AggregateOutType = Z>  >(
	con_rad: f64,

	target_input_pos: &arrayfire::Array<Z>,

	input_pos: &mut arrayfire::Array<Z>,
	input_idx: &mut arrayfire::Array<i32>,
	)
{

	let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);
	let TWO = arrayfire::constant::<f64>(TWO_F64,single_dims).cast::<Z>();




	let input_diff = arrayfire::sub(target_input_pos, input_pos, true);



	let con_rad_sq = con_rad*con_rad;
	let con_rad_sq = arrayfire::constant(con_rad_sq, single_dims).cast::<Z>();

	
	let mut mag2 = arrayfire::pow(&input_diff,&TWO,false);
	mag2 = arrayfire::sum(&mag2, 1);

	//  (con_rad_sq >= mag2 )
	let CMPRET = arrayfire::ge(&con_rad_sq, &mag2, false);
	drop(mag2);

	//Lookup  1 >= dir_line  >= 0
	let idx_intersect = arrayfire::locate(&CMPRET);
	drop(CMPRET);

	*input_pos = arrayfire::lookup(input_pos, &idx_intersect, 0);

	*input_idx = arrayfire::lookup(input_idx, &idx_intersect, 0);


}









pub fn rays_from_neuronsA_to_neuronsB<Z: arrayfire::RealFloating<AggregateOutType = Z>  >(
	con_rad: f64,

	neuronA_pos: &arrayfire::Array<Z>,
	neuronB_pos: &arrayfire::Array<Z>,

	start_line: &mut arrayfire::Array<Z>,
	dir_line: &mut arrayfire::Array<Z>,

	input_idx: &mut arrayfire::Array<i32>,
	hidden_idx: &mut arrayfire::Array<i32>,
	)
{
	let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);
	let TWO = arrayfire::constant::<f64>(TWO_F64,single_dims).cast::<Z>();





	let space_dims: u64 = neuronA_pos.dims()[1];

	let neuronA_num: u64 = neuronA_pos.dims()[0];
	let neuronB_num: u64 = neuronB_pos.dims()[0];

	let tile_dims = arrayfire::Dim4::new(&[neuronB_num,1,1,1]);

	*start_line =  arrayfire::tile(neuronA_pos, tile_dims);


	*dir_line = neuronB_pos.clone();

	
	tileDown(
		neuronA_num,
	
		dir_line
	);


	*dir_line = dir_line.clone() - start_line.clone();


	let con_rad_sq = con_rad*con_rad;
	let con_rad_sq = arrayfire::constant(con_rad_sq, single_dims).cast::<Z>();

	let mut mag2 = arrayfire::pow(dir_line,&TWO,false);
	mag2 = arrayfire::sum(&mag2, 1);

	//  (con_rad_sq >= mag2 )
	let CMPRET = arrayfire::ge(&con_rad_sq, &mag2, false);
	drop(mag2);

	//Lookup  1 >= dir_line  >= 0
	let idx_intersect = arrayfire::locate(&CMPRET);
	drop(CMPRET);

	*start_line = arrayfire::lookup(start_line, &idx_intersect, 0);

	*dir_line = arrayfire::lookup(dir_line, &idx_intersect, 0);

	*input_idx = arrayfire::lookup(input_idx, &idx_intersect, 0);

	*hidden_idx = arrayfire::lookup(hidden_idx, &idx_intersect, 0);

}












