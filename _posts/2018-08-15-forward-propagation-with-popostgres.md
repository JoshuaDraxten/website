---
layout: post
title:  "Implementing Forward Propagation with Postgres"
date:   2018-08-15
categories: projects
written_at: Brooklyn, NY
---

## Why?

Over the last year, I’ve been doing some data analysis and visualization for Property Scout. They are a startup that ——TODO——. Real Estate developers have a litany of metrics they’re interested in about a property, many of which are not easily computed by doing business logic on public records, I tried training a Neural Network on these problems and got good results!

Because Property Scout is dealing with millions of records for every building in NYC, they preferred that the network’s prediction be carried out in their Database environment, Postgres. I did not find any tools on the internet for doing this, so I figured I would document how I did it in case anyone else is in the same boat.

## The good stuff

Convert all your data into an array in a single column. From what I could see, there is no good way to do any matrix-related math in SQL when dealing with multiple columns. You can technically do matrix multiplication and you could probably use `crosstab` to make a table for every record to carry out forward propagation, but that sounds like an absolute nightmare.

**Part 1: Regularize the data**

There are a lot of different things you can do to regularize your data, and I have by no means made a function to do each of these. However, this should give you a good start on how this works.

```sql
CREATE FUNCTION NN_normalize_range( field_value float8, min int, max int ) RETURNS float8[] as $$
BEGIN
	RETURN ARRAY[ 2 * ( (field_value - min)::float / (max - min)::float ) - 1 ];
	END;
$$ LANGUAGE plpgsql;
```

Here we take a number in a field and convert it to an array with a single number in between -1 and 1. The reason it returns an array is because we want to combine all of the data into a single array and this makes it much easier to concatenate into a single array with the operator `||` as per the following example:

```sql
SELECT
	NN_normalize_range( year_built, 1900, 2020 ) ||
	NN_normalize_range( floor_area, 2000, 10000 ) AS data
FROM lots;
```

You can create much more complicated functions by looking  at the documentation ——HERE——. But I’ll give one more example. Lets say I wanted 5 nodes for each borough a building could be in, I would need a function like this:

```sql
-- Necessary for the NN_normalize_option function
CREATE FUNCTION array_search(needle anyelement, haystack anyarray)
RETURNS int AS $$
	SELECT i
	  FROM generate_subscripts($2, 1) AS i
	WHERE $2[i] = $1
  ORDER BY i
$$ LANGUAGE plpgsql;

CREATE FUNCTION NN_normalize_option( field_value varchar, opts varchar[] ) RETURNS float8[] as $$
DECLARE
	output int[] = array_fill(0, ARRAY[array_length(opts, 1) - 1] );
	index int = array_search(field_value, opts);
BEGIN
	IF output[ index ] != null THEN
		output[ array_search(field_value, opts) ] = 1;
	END IF;
	RETURN output;
	END;
$$ LANGUAGE plpgsql;
```

**Part 2: Forward propagation**

Once you have regularized your data you can export it and do your Machine learning magic on it. You will want to export the resulting weight matrices as arrays so we can easily plug it into a Postgres statement (ex: `{ {1,2,3}, {4,5,6} }`).

I used three functions in total to preform forward propagation: sigmoid, addBias and forwardProp:

```sql
CREATE OR REPLACE FUNCTION sigmoid( float8 ) RETURNS float8 as $$
BEGIN
	RETURN 1 / ( 1 + exp( -$1 ) );
	END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION addBias(matrix float8[][]) RETURNS float8[][] as $$
DECLARE
	output float8[][];
	output_row float8[];
	y int; x int;
	row_height int = array_length( matrix, 2 );
	row_width int = array_length( matrix, 1 );
	one float8[] = '{1}';
BEGIN
	output_row = ARRAY[array_fill(1, ARRAY[row_width+2])];
	FOR x in 1..row_height LOOP
		output = array_cat(output, output_row);
	END LOOP;
	FOR y in 1..row_height LOOP
		FOR x in 1..(row_width+1) LOOP
			output[y][x+1] = matrix[y][x];
		END LOOP;
	END LOOP;
	RETURN output;
	END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION forwardProp(A float8[][], B float8[][]) RETURNS float8[][] as $$
DECLARE
	i int; j int; x int;
	sum float8;
	n int = array_length( A, 1 );
	m int = array_length( A, 2 );
	p int = array_length( B, 2 );
	C float8[][];
	C_row float8[][];
	A_bias float8[][] = addBias( A );
BEGIN
	C_row = ARRAY[array_fill(0, ARRAY[p])];
	FOR x in 1..n LOOP
		C = array_cat(C, C_row);
	END LOOP;
	FOR i in 1..n LOOP
		FOR j in 1..p LOOP
			sum = 0;
			FOR k in 1..m LOOP
				sum = sum + (A[i][k] * B[k][j]);
			END LOOP;
			C[i][j] = sigmoid(sum);
		END LOOP;
	END LOOP;
	RETURN C;
	END;
$$ LANGUAGE plpgsql;
```

If you familiarize yourself with plpgsql, it is surprisingly straightforward to follow along with the above.

**Part 3: Putting it all together**

Let’s pretend we had a Neural Network with a single hidden layer that we trained to look at the year a building was built and the floor area of a condo within it and it’s goal was to predict the condo’s price. You can implement this simply with the following code:

```sql
SELECT
	forwardProp(
		-- All these functions deal with two-dimentional arrays
		-- so we convert our data to fit that
		ARRAY[
			NN_normalize_range( year_built, 1900, 2020 ) ||
			NN_normalize_range( floor_area, 2000, 10000 ) AS data
		],
		 -- Weights computed with back propagation.
		'{ {0.0995198}, {0.1428894} }'
	) AS prediction
FROM lots;
```
