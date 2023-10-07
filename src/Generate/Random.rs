use arrayfire;


use rayon::prelude::*;


use std::collections::HashMap;


const TWO_F64: f64 = 2.0;
const ONE_F64: f64 = 1.0;
const ZERO_F64: f64 = 0.0;


const NEGATIVE_ONE_F64: f64 = -1.0;

const EPSILON2_F64: f64 = 1.0e-8;




pub fn generate_random_rays_to_center<Z: arrayfire::RealFloating<AggregateOutType = Z>  >(
	neuron_pos: &arrayfire::Array<Z>,
	ray_num: u64,
	con_rad: f64,

	start_line: &mut arrayfire::Array<Z>,
	dir_line: &mut arrayfire::Array<Z>
	)
{

	let single_dims = arrayfire::Dim4::new(&[1,1,1,1]);
	let TWO = arrayfire::constant::<f64>(TWO_F64,single_dims).cast::<Z>();

	let NEGATIVE_ONE = arrayfire::constant::<f64>(NEGATIVE_ONE_F64,single_dims).cast::<Z>();


	let EPSILON2 = arrayfire::constant::<f64>(EPSILON2_F64,single_dims).cast::<Z>();








	let space_dims: u64 = neuron_pos.dims()[1];



	let tile_dims = arrayfire::Dim4::new(&[ray_num,1,1,1]);

	*start_line =  arrayfire::tile(neuron_pos, tile_dims);

	*dir_line =  start_line.clone()*NEGATIVE_ONE;




	//Mag of dir_line
	let mut mag2 = arrayfire::pow(dir_line,&TWO,false);
	mag2 = arrayfire::sum(&mag2, 1);




	//Generate random vectors
	let start_line_num =  start_line.dims()[0];
	let rand_dims = arrayfire::Dim4::new(&[start_line_num,space_dims,1,1]);
	let mut rand_vec = (arrayfire::randu::<f64>(rand_dims) - 0.5f64);
	
	//Normalize random Vector
	let mut mag = arrayfire::pow(&rand_vec,&TWO,false);
	mag = arrayfire::sum(&mag, 1);
	mag = arrayfire::sqrt(&mag) + epsilon2;

	
	//Scale random vector to connection radius
	rand_vec = arrayfire::div(&rand_vec,&mag,true);
	mag = arrayfire::sqrt(&mag2);
	rand_vec = arrayfire::mul(&rand_vec, &mag, true);
	drop(mag);





	//Vector Projection
	let mut projvec = arrayfire::mul(&rand_vec, dir_line, false);
	projvec = arrayfire::sum(&projvec, 1);

	mag2 = mag2 + epsilon2;
	projvec = arrayfire::div(&projvec, &mag2, false);
	drop(mag2);

	//Vector rejection
	projvec = rand_vec.clone() -  arrayfire::mul(&projvec, dir_line,true);
	drop(rand_vec);

	//Random scale
	let rand2_dims = arrayfire::Dim4::new(&[start_line_num,1,1,1]);
	let mut rand2_vec = 2.0f64*arrayfire::randu::<f64>(rand2_dims) ;
	projvec = arrayfire::mul(&projvec, &rand2_vec, true);

	*dir_line = dir_line.clone() + projvec;
	


	//Scale dir line
	let mut mag3 = arrayfire::pow(dir_line ,&TWO,false);
	mag3 = arrayfire::sum(&mag3, 1);
	mag3 = arrayfire::sqrt(&mag3) + epsilon2;

	*dir_line = con_rad*arrayfire::div(dir_line, &mag3, true);



}











