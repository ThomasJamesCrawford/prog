/*************************************************************
*
*         Author: Thomas Crawford 21486482
*		  CITS3402 Assignment2
*		  
*		  Adapted Heavily from Matrix Multiplication Example
*		  http://teaching.csse.uwa.edu.au/units/CITS3402/lectures/mpi-slides.pdf
*
*/
#define _CRT_SECURE_NO_DEPRECATE

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#define MASTER 0 /* taskid of first task */
#define FROM_MASTER 1 /* setting a message type */
#define FROM_WORKER 2 /* setting a message type */

#define INT_MAX 2147483647

void freeArr(int** arr);
int** allocArr(int rows, int cols);
int** readGraph(char* filename, int* v);
int* dijkstra(int** graph, int start, int v);
int getMin(int* dist, int* visited, int v);
void writeToFile(int** graph, char* filename, int size);

int main(int argc, char* argv[])
{
	int numtasks, /* number of tasks in partition */
		taskid, /* a task identifier */
		numworkers, /* number of worker tasks */
		source, /* task id of message source */
		dest, /* task id of message destination */
		mtype, /* message type */
		rows, /* rows for each worker */
		averow, extra, offset, /* used to determine rows sent to each worker */
		rc, v, i, j, /* misc */
		** graph, /* input graph[][] */
		** distance; /* output distances[][]*/

	MPI_Status status;

	rc = MPI_Init(&argc, &argv);

	// must have at least 1 additional arg
	if (argc < 2)
	{
		printf("No input file given. Quitting..\n");
		MPI_Abort(MPI_COMM_WORLD, 0);
		exit(1);
	}

	MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

	/**************************** master task ************************************/
	if (taskid == MASTER)
	{
		printf("Project2 has started with %d tasks.\n", numtasks);

		/* Create the graph */
		char* filename = argv[argc - 1]; // will always be last arg
		graph = readGraph(filename, &v);

		if (numtasks > 1)
		{
			numworkers = numtasks - 1;

			/* Calculate work required for each worker */
			averow = v / numworkers;
			extra = v % numworkers;
		}
		else 
		{
			numworkers = 0; // master does work
			/* Calculate work required for each worker */
			averow = v;
			extra = 0;
		}


		offset = 0;

		distance = allocArr(v, v);

		clock_t begin, end;
		begin = clock();

		/* Send each worker the rows of graph they are responsible for; 
		*  Each row of the graph is equivalent to an iteration of dijkstras with row index as source
		*/
		mtype = FROM_MASTER;
		for (dest = 1; dest <= numworkers; dest++)
		{
			rows = (dest <= extra) ? averow + 1 : averow;

			printf("Sending %d rows to task %d offset=%d\n",
				rows, dest, offset);

			/* Send dimension of graph */
			MPI_Send(&v, 1, MPI_INT, dest,
				mtype, MPI_COMM_WORLD);

			/* Send index to start at */
			MPI_Send(&offset, 1, MPI_INT, dest,
				mtype, MPI_COMM_WORLD);

			/* Send amount to do */
			MPI_Send(&rows, 1, MPI_INT, dest,
				mtype, MPI_COMM_WORLD);

			if (rows > 0) // slave will die when it gets 0 rows
			{
				/* Send graph */
				MPI_Send(&graph[0][0], v * v, MPI_INT,
					dest, mtype, MPI_COMM_WORLD);
			}

			offset = offset + rows;
		}

		/* If master is the only one available*/
		if (numworkers == 0)
		{
			printf("Performing %d rows on master offset=%d\n", averow, offset);
			for (i = 0; i < averow; i++)
			{
				int* distances = dijkstra(graph, offset + i, v);

				for (j = 0; j < v; j++)
				{
					distance[offset + i][j] = distances[j];
				}

				free(distances);
			}
		}

		// dont wait for workers who got sent 0 rows
		if (v < numworkers)
		{
			printf("Not waiting for %d workers\n", numworkers-v);
			numworkers = v;
		}

		/* Receive results from worker tasks */
		mtype = FROM_WORKER;
		for (i = 1; i <= numworkers; i++)
		{
			source = i;

			/* Receive start of work */
			MPI_Recv(&offset, 1, MPI_INT, source, mtype,
				MPI_COMM_WORLD, &status);

			/* Receive number of work completed */
			MPI_Recv(&rows, 1, MPI_INT, source, mtype,
				MPI_COMM_WORLD, &status);

			/* Receive distances result of work and populate distances */
			MPI_Recv(&distance[offset][0], rows * v, MPI_INT,
				source, mtype, MPI_COMM_WORLD, &status);

			printf("Received %d rows from task %d\n", rows, source);
		}

		end = clock();
		double time = (double)((double)(end - begin) / CLOCKS_PER_SEC);
		printf("%fs\n", time);

		writeToFile(graph, filename, v);

		printf("Done.\n");

		/* Free master memory */
		freeArr(graph);
		freeArr(distance);
	}

	/**************************** worker task ************************************/
	if (taskid > MASTER)
	{
		mtype = FROM_MASTER;

		/* Receive size of the graph */
		MPI_Recv(&v, 1, MPI_INT, MASTER, mtype,
			MPI_COMM_WORLD, &status);

		/* Receive index of work completed */
		MPI_Recv(&offset, 1, MPI_INT, MASTER, mtype,
			MPI_COMM_WORLD, &status);

		/* Receive number of tasks to complete */
		MPI_Recv(&rows, 1, MPI_INT, MASTER,
			mtype, MPI_COMM_WORLD, &status);

		if (rows <= 0) 
		{
			// no need to do anything
			MPI_Finalize();
			return 0;
		}

		// allocate graph
		graph = allocArr(v, v);

		/* Receive the graph */
		MPI_Recv(&graph[0][0], v * v, MPI_INT, MASTER,
			mtype, MPI_COMM_WORLD, &status);

		// make results array
		int** results = allocArr(rows, v);

		// do work
		for (i = 0; i < rows; i++)
		{
			int* distances = dijkstra(graph, offset + i, v);

			// read result into sequentially allocated array
			for (j = 0; j < v; j++)
			{
				results[i][j] = distances[j];
			}

			free(distances);
		}

		mtype = FROM_WORKER;

		/* Send start index of work completed back */
		MPI_Send(&offset, 1, MPI_INT, MASTER, mtype,
			MPI_COMM_WORLD);

		/* Send amount of work completed back */
		MPI_Send(&rows, 1, MPI_INT, MASTER, mtype,
			MPI_COMM_WORLD);

		/* Send results of work completed back */
		MPI_Send(&results[0][0], rows * v, MPI_INT, MASTER,
			mtype, MPI_COMM_WORLD);

		/* free workers memory - already sent results */
		freeArr(results);
		freeArr(graph);
	}

	MPI_Finalize();
}

/* return index of the minimum in dist thats univisited in visited */
int getMin(int* dist, int* visited, int v) {
	int index = 0;
	int min = INT_MAX;

	int i;
	for (i = 0; i < v; i++) {
		if (visited[i] == 0 && dist[i] <= min) {
			min = dist[i];
			index = i;
		}
	}

	return index;
}

/* dijkstra algorithm 
* adapted from
* https://www.geeksforgeeks.org/c-program-for-dijkstras-shortest-path-algorithm-greedy-algo-7/
*/
int* dijkstra(int** graph, int start, int v) {
	int* dist;
	int* visited;

	dist = (int *)malloc(sizeof(int) * v);
	visited = (int*)malloc(sizeof(int) * v);

	int i;
	for (i = 0; i < v; i++) {
		dist[i] = INT_MAX;
		visited[i] = 0;
	}

	dist[start] = 0;

	int x, y;
	for (x = 0; x < v - 1; x++) {
		int u = getMin(dist, visited, v);
		visited[u] = 1;

		for (y = 0; y < v; y++) {
			if (visited[y] != 1 &&
				graph[u][y] > 0 &&
				dist[u] != INT_MAX &&
				dist[u] + graph[u][y] < dist[y]) {
				dist[y] = dist[u] + graph[u][y];
			}
		}
	}

	free(visited);
	return dist;
}

/* continuous alloc
* https://stackoverflow.com/questions/5901476/sending-and-receiving-2d-array-over-mpi
*/
int** allocArr(int rows, int cols) {
	int* data = (int*)malloc(rows * cols * sizeof(int));
	int** array = (int**)malloc(rows * sizeof(int*));

	int i;
	for (i = 0; i < rows; i++)
		array[i] = &(data[cols * i]);

	return array;
}

/*
* free int** array
*/
void freeArr(int** arr)
{
	if (arr[0])
	{
		free(arr[0]);
	}

	if (arr)
	{
		free(arr);
	}
}

int** readGraph(char* filename, int* v) 
{
	int** graph;
	int size; 

	FILE* f;
	f = fopen(filename, "rb");

	if (f == NULL)
	{
		printf("Invalid filename : %s\n", filename);
		printf("Exiting...\n");
		MPI_Abort(MPI_COMM_WORLD, 0);
		exit(1);
	}

	fread(&size, sizeof(int), 1, f);

	graph = allocArr(size, size);

	int i, j;
	for (i = 0; i < size; i++)
	{
		for (j = 0; j < size; j++)
		{
			fread(&graph[i][j], sizeof(int), 1, f);
		}
	}

	fclose(f);

	*v = size;
	return graph;
}

void writeToFile(int** graph, char* filename, int size)
{
	int len = strlen(filename);
	char newFilename[256];

	if (len > 256)
	{
		printf("Filename over 256 characters : %s\n", filename);
		printf("Exiting...\n");
		MPI_Abort(MPI_COMM_WORLD, 0);
		exit(1);
	}

	strcpy(newFilename, filename);

	newFilename[len - 1 - 1] = 'o';
	newFilename[len - 1] = 'u';
	newFilename[len] = 't';
	newFilename[len + 1] = '\0';

	printf("%s\n", newFilename);

	FILE* f;
	f = fopen(newFilename, "wb+");

	if (f == NULL)
	{
		printf("Couldn't create file : %s\n", filename);
		printf("Exiting...\n");
		MPI_Abort(MPI_COMM_WORLD, 0);
		exit(1);
	}

	fwrite(&size, sizeof(int), 1, f);
	fwrite(graph, sizeof(int), size * size, f);

	fclose(f);
	printf("Results written to binary file : %s\n", newFilename);
}
