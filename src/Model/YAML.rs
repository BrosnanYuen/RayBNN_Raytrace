use arrayfire;


use rayon::prelude::*;


use std::fs;

use std::collections::HashMap;

use std::fs::File;
use std::io::Write;



use std::io::{self, BufRead};
use std::path::Path;


fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where P: AsRef<Path>, {
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}






pub fn read(
	filename: &str,

    modeldata_string: &mut HashMap<String, String>,
	modeldata_float: &mut HashMap<String, f64>,
    modeldata_int: &mut HashMap<String, u64>,
	)
{

	if let Ok(lines) = read_lines(filename) {
        // Consumes the iterator, returns an (Optional) String
        for line in lines {
            if let Ok(data) = line {
				if data.contains("#")
				{
					continue;
				}

				if data.contains(":") == false
				{
					continue;
				}

				if data.contains("'")
				{
					let datasplit: Vec<&str> = data.split(":").collect();
					let key = datasplit[0].clone().to_string();

					let mut value = datasplit[1].clone().to_string();

					let datasplit: Vec<&str> = value.split("'").collect();

					value = datasplit[1].clone().to_string();

					//println!("key V{}V",key);
					//println!("value V{}V",value);
					modeldata_string.insert(key.clone(), value.clone());
				}
				else if data.contains(".")
				{
					let datasplit: Vec<&str> = data.split(":").collect();
					let key = datasplit[0].clone().to_string();

					let mut value = datasplit[1].clone().to_string();
					value = value.replace(" ", "");

					let value = value.parse::<f64>().unwrap();

					modeldata_float.insert(key.clone(), value.clone());
				}
				else 
				{
					let datasplit: Vec<&str> = data.split(":").collect();
					let key = datasplit[0].clone().to_string();

					let mut value = datasplit[1].clone().to_string();
					value = value.replace(" ", "");

					let value = value.parse::<u64>().unwrap();

					modeldata_int.insert(key.clone(), value.clone());
				}


            }
        }
    }

}





pub fn write(
	filename: &str,

    modeldata_string: &HashMap<String, String>,
	modeldata_float: &HashMap<String, f64>,
    modeldata_int: &HashMap<String, u64>,
	)
{
	let mut strvec: Vec<String> = Vec::new();

	for (key, value) in modeldata_int {
		let tmp = format!("{}: {}\n", key.clone(), value.clone());
		strvec.push(tmp.clone());
	}

	for (key, value) in modeldata_float {
		let tmp = format!("{}: {}\n", key.clone(), value.clone());
		strvec.push(tmp.clone());
	}

	for (key, value) in modeldata_string {
		let tmp = format!("{}: '{}'\n", key.clone(), value.clone());
		strvec.push(tmp.clone());
	}

	let tmpstr: String = strvec.into_par_iter().collect::<String>();


	let mut file0 = File::create(filename).unwrap();
	writeln!(file0, "{}", tmpstr);
}


