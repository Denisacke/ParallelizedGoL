#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ALIVE 1
#define DEAD 0
#define NUM_ITERATIONS 1

int **allocate_grid(int rows, int cols);

void free_grid(int **grid);

void initialize_grid(int **grid, int rows, int cols);

void exchange_borders(int **grid, int rows, int cols, int rank, int size);

void compute_next_state(int **grid, int **new_grid, int rows, int cols);

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows = 10, cols = 10; // Total grid size
    int local_rows = rows / size; // Rows handled by each process

    if (rows % size != 0) {
        if (rank == 0)
            printf("Rows must be evenly divisible by the number of processes.\n");
        MPI_Finalize();
        return 1;
    }

    // Allocate grid (including ghost rows)
    int **grid = allocate_grid(local_rows + 2, cols);
    int **new_grid = allocate_grid(local_rows + 2, cols);

    srand(time(nullptr) + rank);
    initialize_grid(grid, local_rows + 2, cols);

    for (int t = 0; t <= NUM_ITERATIONS; t++) {
        exchange_borders(grid, local_rows + 2, cols, rank, size);
        compute_next_state(grid, new_grid, local_rows + 2, cols);

        // Swap grids
        int **temp = grid;
        grid = new_grid;
        new_grid = temp;

        // Gather the grids at rank 0
        int* gathered_grid = NULL;
        if (rank == 0) {
            gathered_grid = (int*)malloc(rows * cols * sizeof(int));
        }

        // Send local grid (excluding ghost rows) to rank 0
        MPI_Gather(grid[1], local_rows * cols, MPI_INT,
                   gathered_grid, local_rows * cols, MPI_INT,
                   0, MPI_COMM_WORLD);

        // Rank 0 prints the grid
        if (rank == 0) {
            printf("Iteration %d:\n", t);
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    printf("%c", gathered_grid[i * cols + j] == ALIVE ? 'O' : '.');
                }
                printf("\n");
            }
            printf("\n");
            free(gathered_grid);
        }

    }

    free_grid(grid);
    free_grid(new_grid);

    MPI_Finalize();
    return 0;
}

// Allocate a 2D grid dynamically
int **allocate_grid(int rows, int cols) {
    int **grid = (int **) malloc(rows * sizeof(int *));
    grid[0] = (int *) malloc(rows * cols * sizeof(int));
    for (int i = 1; i < rows; i++) {
        grid[i] = grid[0] + i * cols;
    }
    return grid;
}

// Free a dynamically allocated grid
void free_grid(int **grid) {
    free(grid[0]);
    free(grid);
}

// Initialize the grid with random values
void initialize_grid(int **grid, int rows, int cols) {
    for (int i = 1; i < rows - 1; i++) { // Exclude ghost rows
        for (int j = 0; j < cols; j++) {
            grid[i][j] = rand() % 2; // Randomly ALIVE or DEAD
        }
    }

    // Set ghost rows to DEAD
    for (int j = 0; j < cols; j++) {
        grid[0][j] = DEAD;
        grid[rows - 1][j] = DEAD;
    }
}

// Exchange ghost rows with neighboring processes
void exchange_borders(int **grid, int rows, int cols, int rank, int size) {
    int above = rank - 1;
    int below = rank + 1;
    MPI_Request request[4];
    MPI_Status status[4];

    // Asynchronous send/receive
    if (below < size) {
        MPI_Isend(grid[rows - 2], cols, MPI_INT, below, 0, MPI_COMM_WORLD, &request[0]);
        MPI_Irecv(grid[rows - 1], cols, MPI_INT, below, 0, MPI_COMM_WORLD, &request[1]);
    }

    if (above >= 0) {
        MPI_Isend(grid[1], cols, MPI_INT, above, 0, MPI_COMM_WORLD, &request[2]);
        MPI_Irecv(grid[0], cols, MPI_INT, above, 0, MPI_COMM_WORLD, &request[3]);
    }

    MPI_Waitall(4, request, status);

}

void compute_next_state(int **grid, int **new_grid, int rows, int cols) {
    for (int i = 1; i < rows - 1; i++) { // Exclude ghost rows
        for (int j = 0; j < cols; j++) {
            int alive_neighbors = 0;

            for (int di = -1; di <= 1; di++) {
                for (int dj = -1; dj <= 1; dj++) {
                    if (di == 0 && dj == 0) continue; // Skip self
                    int ni = i + di;
                    int nj = j + dj;

                    if (nj >= 0 && nj < cols) {
                        alive_neighbors += grid[ni][nj];
                    }
                }
            }

            if (grid[i][j] == ALIVE) {
                new_grid[i][j] = (alive_neighbors == 2 || alive_neighbors == 3) ? ALIVE : DEAD;
            } else {
                new_grid[i][j] = (alive_neighbors == 3) ? ALIVE : DEAD;
            }
        }
    }
}
