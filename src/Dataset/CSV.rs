use arrayfire;


use rayon::prelude::*;


use std::fs;

use std::collections::HashMap;

use std::fs::File;
use std::io::Write;


use nohash_hasher;




use crate::Dataset::ParseString::str_to_vec_cpu as str_to_vec_cpu;
use crate::Dataset::ParseString::vec_cpu_to_str as vec_cpu_to_str;



pub fn file_to_vec_cpu<Z: std::str::FromStr + Send + Sync>(
	filename: &str
) -> (Vec<Z>, HashMap<String,u64>)  {

    let mut metadata = HashMap::new();


	let contents = fs::read_to_string(filename).expect("error");

	let tmp = contents.par_split('\n').map(str_to_vec_cpu );

    metadata.insert("dims".to_string(), 2);

    let dim0 = (tmp.clone().count() as u64) - 1;
    metadata.insert("dim0".to_string(), dim0.clone());

    let result: Vec<Z> = tmp.flatten_iter().collect();

    let dim1 = (result.len() as u64)/dim0;

    metadata.insert("dim1".to_string(), dim1.clone());
    
    (result,metadata)
}




pub fn file_to_arrayfire<Z: std::str::FromStr + arrayfire::HasAfEnum + Send + Sync>(
	filename: &str,
	) -> arrayfire::Array<Z>  {

	let (vector,metadata) = file_to_vec_cpu::<Z>(filename);

    let dim0 = metadata["dim0"];
    let dim1 = metadata["dim1"];

	let arr_dims = arrayfire::Dim4::new(&[dim1, dim0, 1, 1]);
	let outarr = arrayfire::Array::new(&vector, arr_dims);


	arrayfire::transpose(&outarr,false)
}




pub fn write_vec_cpu_to_csv<Z: arrayfire::HasAfEnum + Sync + Send>(
	filename: &str,
	invec: &Vec<Z>,
	metadata: &HashMap<String,u64>,
	)
{

	let dim0 = metadata["dim0"];
    let dim1 = metadata["dim1"];

	//let mut wtr0 = vec_cpu_to_str::<Z>(invec);
	let mut tmp: String = invec.par_chunks_exact(dim1 as usize).map(vec_cpu_to_str ).map(|x| x+"\n").collect();
	tmp.pop();


	let mut file0 = File::create(filename).unwrap();
	writeln!(file0, "{}", tmp);
}





pub fn write_arrayfire_to_csv<Z: arrayfire::HasAfEnum + Sync + Send>(
	filename: &str,
	arr: &arrayfire::Array<Z>
	)
{

	let mut metadata: HashMap<String,u64> = HashMap::new();

	metadata.insert("dim0".to_string(), arr.dims()[0]);
    metadata.insert("dim1".to_string(), arr.dims()[1]);


	let tmp = arrayfire::transpose(arr, false);

	let mut invec = vec!(Z::default();tmp.elements());
	tmp.host(&mut invec);

	write_vec_cpu_to_csv::<Z>(
		filename,
		&invec,
		&metadata
	);
}










pub fn file_to_hash_cpu<Z: std::str::FromStr + Send + Sync + Clone>(
	filename: &str,
	sample_size: u64,
	batch_size: u64
	) -> (nohash_hasher::IntMap<u64, Vec<Z> >, HashMap<String,u64>)  {

	
	

	let (arr,metadata) = file_to_vec_cpu(filename);

	let arr_size = arr.len() as u64;
	let item_num = (arr_size/(sample_size*batch_size));

	let mut lookup: nohash_hasher::IntMap<u64, Vec<Z> >  = nohash_hasher::IntMap::default();
	let mut start:usize = 0;
	let mut end:usize = 0;
	for i in 0..item_num
	{
		start = (i*(sample_size*batch_size)) as usize;
		end = ((i+1)*(sample_size*batch_size)) as usize;
		lookup.insert(i, (arr[start..end]).to_vec() );
	}

	(lookup,metadata)
}








pub fn file_to_hash_arrayfire<Z: std::str::FromStr + arrayfire::HasAfEnum + Send + Sync + Clone>(
	filename: &str,
	dims: arrayfire::Dim4
	) -> (nohash_hasher::IntMap<u64, arrayfire::Array<Z>  >, HashMap<String,u64>)  {


	let mut lookup2: nohash_hasher::IntMap<u64, arrayfire::Array<Z>  >   = nohash_hasher::IntMap::default();

	let (lookup, metadata) = file_to_hash_cpu(
		filename,
		dims[1],
		dims[0]
	);

	let item_num = lookup.len() as u64;

    let arr_dims = arrayfire::Dim4::new(&[dims[1], dims[0], 1, 1]);
	for i in 0..item_num
	{
		let mut temparr = arrayfire::Array::new(&lookup[&i], arr_dims);

		temparr = arrayfire::transpose(&temparr,false);

		lookup2.insert(i, temparr);
	}


	(lookup2,metadata)
}





