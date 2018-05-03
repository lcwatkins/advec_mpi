
#include <mpi.h>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <math.h>
#include <assert.h>

// define new type: matrix of doubles
typedef std::vector<std::vector<double> > dmatrix ;

double **alloc2darray(int size);
void init(int N, double dx, dmatrix& C);
void init_mpi(int N, double dx, double **C, int subgridlen, int* coords);
void exchangeGhostCells(double **my_C, int subgridlen, MPI_Comm cartcomm, int* nbrs, int* coords, int mype, int blockflag);
void printMat(int N, dmatrix& C);
void copyAtoB(dmatrix& A, dmatrix& B);
void update(dmatrix& C, dmatrix& C_old, int N, double h, double u, double v);
void update_mpi(dmatrix& C, dmatrix& C_old, int n, double h, double u, double v);
void printToFile(int N, int step, double **A);
void printToFile_mpi(int step, double **C, int N, int subgridlen, int ngrid, int myrank);

int main(int argc, char *argv[]){
    int rflag;
    rflag = 0;
    // parameters: read in
    double NT, T, L, u, v, dx, dt, h;
    int N, nthreads, blockflag;
    std::string mpimode;
    if (rflag == 1){
        if (argc != 9) {
            printf("Should have 8 args\n");
            return 0;
        }
        N = std::stoi(argv[1]);
        NT = std::stof(argv[2]);
        L = std::stof(argv[3]);
        T = std::stof(argv[4]);
        u = std::stof(argv[5]);
        v = std::stof(argv[6]);
        nthreads = std::stoi(argv[7]);
        mpimode = argv[8];
    } else {
        N = 16;
        NT = 20000.0;
        L = 1.0;
        T = 1.0e6;
        u = 5.0e-7;
        v = 2.85e-7;
        nthreads = 1;
        mpimode = "mpi_non_blocking";
    }
    //printf("N: %d \nNT: %f \nL: %f \nT: %f \nu: %f \nv: %f \n", N, NT, L, T, u, v);
    dx = L/N;
    dt = T/NT;
    h = dx/sqrt(2*(u*u+v*v));
    if (dt>h){
        printf("UNSTABLE! h: %f \ndt: %f\n",h,dt);
    }
    h = dt/(2*dx);

    printf("N: %d, dx: %f\n", N, dx);

    // SETUP MPI STUFF
    if (mpimode.compare("mpi_non_blocking") == 0) {
        printf("non-blocking mode\n");
        blockflag = 0;
    } else if (mpimode.compare("mpi_blocking") == 0){
        printf("blocking mode\n");
        blockflag = 1;
    } else if (mpimode.compare("hybrid") == 0) {
        printf("hybrid mode\n");
    } else {
        printf("non valid MPI mode\n");
        return 1;
    }

    int nprocs, stat, mype, ngrid, subgridlen;
    MPI_Init(&argc, &argv);
    stat = MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    assert(stat == MPI_SUCCESS);
    stat = MPI_Comm_rank(MPI_COMM_WORLD, &mype);
    assert(stat == MPI_SUCCESS);
    assert(fmod(N, sqrt(nprocs)) == 0);

    // setup cartesian communicator 
    MPI_Comm cartcomm;
    int ndim = 2;
    int dims[2] = {0,0};
    int periods[2] = {1,1};
    MPI_Dims_create(nprocs, ndim, dims);
    MPI_Cart_create(MPI_COMM_WORLD, ndim, dims, periods, 0, &cartcomm);
   
    // assuming (as told we could) that we have a square num of procs 
    // so our virtual layout is ngrid x ngrid procs, each proc has
    // a subgridlen x subgridlen array portion of whole matrix C
    ngrid = sqrt(nprocs);
    subgridlen = N/ngrid;

    // get coords of the subgrid on each proc
    int coords[ndim] ;
    stat = MPI_Cart_coords(cartcomm, mype, ndim, coords);
    printf("%d: (%d,%d)\n",mype,coords[0],coords[1]);
    
    // initialize matrix portion on each proc
    // ghost cells included in C, so C is size subgridlen+2 
    // cells "owned" by C are in the interior
    double **my_C;
    double **my_C_old;
    my_C = alloc2darray(subgridlen+2);
    my_C_old = alloc2darray(subgridlen+2);
    init_mpi(N, dx, my_C, subgridlen, coords);
    init_mpi(N, dx, my_C_old, subgridlen, coords);


    // get neighbors: nbrs: (up,down,left,right)
    int nbrs[4];
    stat = MPI_Cart_shift(cartcomm, 0, 1, &nbrs[0], &nbrs[1]);
    stat = MPI_Cart_shift(cartcomm, 1, 1, &nbrs[2], &nbrs[3]);
   // printf("proc %d, up: %d, down: %d, left: %d, right: %d\n",mype, nbrs[0], nbrs[1], nbrs[2], nbrs[3]);

    printToFile_mpi(100, my_C, N, subgridlen, ngrid, mype);
    MPI_Barrier(cartcomm);


    NT = 1;
    // run for NT timesteps
    for (int step=0;step<NT;step++){
        exchangeGhostCells(my_C_old, subgridlen, cartcomm, nbrs, coords, mype, blockflag);
        update_mpi(my_C, my_C_old, subgridlen, h, u, v);
        copyAtoB(my_C,my_C_old);
    //    if (step%10 == 0){
    //        printToFile_mpi(step, my_C, N, subgridlen, ngrid, mype);
    //    }
    }
    printToFile_mpi(101, my_C, N, subgridlen, ngrid, mype);
    MPI_Finalize();
    return 0;
}

void exchangeGhostCells(double **my_C, int subgridlen, MPI_Comm cartcomm, int* nbrs, int* coords, int mype, int blockflag){
    int stat;
    MPI_Status status;
    MPI_Datatype mpi_column;
    MPI_Type_vector(subgridlen+2, 1, subgridlen+2, MPI_DOUBLE, &mpi_column);
    MPI_Type_commit(&mpi_column);

    // mpi tags:    23 == left->right
    //              32 == right->left
    //              01 == up->down
    //              10 == down->up

    if (blockflag == 1) {
        printf("BLOCK\n");
        // blocking send/recv
        if (coords[1] == 0) {
            // SEND RIGHT SIDE TO RIGHT NEGIHBOR 
            MPI_Send(&(my_C[0][subgridlen]), 1, mpi_column, nbrs[3], 23, MPI_COMM_WORLD);
            //printf("%d waiting for %d\n",mype,nbrs[2]);
            MPI_Recv(&(my_C[0][0]), 1, mpi_column, nbrs[2], 23, MPI_COMM_WORLD, &status);
            assert(stat == MPI_SUCCESS);
            //printf("%d received from %d\n",mype,nbrs[2]);

            // SEND LEFT SIDE TO LEFT NEGIHBOR 
            MPI_Send(&(my_C[0][1]), 1, mpi_column, nbrs[2], 32, MPI_COMM_WORLD);
            //printf("%d waiting for %d\n",mype,nbrs[3]);
            MPI_Recv(&(my_C[0][subgridlen+1]), 1, mpi_column, nbrs[3], 32, MPI_COMM_WORLD, &status);
            assert(stat == MPI_SUCCESS);
            //printf("%d received from %d\n",mype,nbrs[3]);
        } else {
            // SEND RIGHT SIDE TO RIGHT NEGIHBOR 
            //printf("%d waiting for %d\n",mype,nbrs[2]);
            MPI_Recv(&(my_C[0][0]), 1, mpi_column, nbrs[2], 23, MPI_COMM_WORLD, &status);
            assert(stat == MPI_SUCCESS);
            //printf("%d received from %d\n",mype,nbrs[2]);
            MPI_Send(&(my_C[0][subgridlen]), 1, mpi_column, nbrs[3], 23, MPI_COMM_WORLD);

            // SEND LEFT SIDE TO LEFT NEGIHBOR 
            //printf("%d waiting for %d\n",mype,nbrs[3]);
            MPI_Recv(&(my_C[0][subgridlen+1]), 1, mpi_column, nbrs[3], 32, MPI_COMM_WORLD, &status);
            assert(stat == MPI_SUCCESS);
            //printf("%d received from %d\n",mype,nbrs[3]);
            MPI_Send(&(my_C[0][1]), 1, mpi_column, nbrs[2], 32, MPI_COMM_WORLD);
        }
        if (coords[0] == 0){
            // SEND TOP TO ABOVE NEIGHBOR
            MPI_Send(&(my_C[1][0]), subgridlen+2, MPI_DOUBLE, nbrs[0], 10, MPI_COMM_WORLD);
            //printf("%d waiting for %d\n",mype,nbrs[1]);
            MPI_Recv(&(my_C[subgridlen+1][0]), subgridlen+2, MPI_DOUBLE, nbrs[1], 10, MPI_COMM_WORLD, &status);
            assert(stat == MPI_SUCCESS);
            //printf("%d received from %d\n",mype,nbrs[1]);

            // SEND BOTTOM TO BELOW NEIGHBOR
            MPI_Send(&(my_C[subgridlen][0]), subgridlen+2, MPI_DOUBLE, nbrs[1], 01, MPI_COMM_WORLD);
            //printf("%d waiting for %d\n",mype,nbrs[0]);
            MPI_Recv(&(my_C[0][0]), subgridlen+2, MPI_DOUBLE, nbrs[0], 01, MPI_COMM_WORLD, &status);
            assert(stat == MPI_SUCCESS);
            //printf("%d received from %d\n",mype,nbrs[0]);
        } else {
            //printf("%d waiting for %d\n",mype,nbrs[1]);
            MPI_Recv(&(my_C[subgridlen+1][0]), subgridlen+2, MPI_DOUBLE, nbrs[1], 10, MPI_COMM_WORLD, &status);
            assert(stat == MPI_SUCCESS);
            //printf("%d received from %d\n",mype,nbrs[1]);
            MPI_Send(&(my_C[1][0]), subgridlen+2, MPI_DOUBLE, nbrs[0], 10, MPI_COMM_WORLD);

            //printf("%d waiting for %d\n",mype,nbrs[0]);
            MPI_Recv(&(my_C[0][0]), subgridlen+2, MPI_DOUBLE, nbrs[0], 01, MPI_COMM_WORLD, &status);
            assert(stat == MPI_SUCCESS);
            //printf("%d received from %d\n",mype,nbrs[1]);
            MPI_Send(&(my_C[subgridlen][0]), subgridlen+2, MPI_DOUBLE, nbrs[1], 01, MPI_COMM_WORLD);
        }
    } else {
        // non-blocking send/recv
        printf("NONBLOCK\n");
        MPI_Status status[4*2];
        MPI_Request reqs[4*2];
        MPI_Irecv(&(my_C[0][0]),1,mpi_column, nbrs[2], 23, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(&(my_C[0][subgridlen+1]),1,mpi_column, nbrs[3], 32, MPI_COMM_WORLD, &reqs[1]);
        MPI_Irecv(&(my_C[0][0]),subgridlen+2, MPI_DOUBLE, nbrs[0], 01, MPI_COMM_WORLD, &reqs[2]);
        MPI_Irecv(&(my_C[subgridlen+1][0]),subgridlen+2, MPI_DOUBLE, nbrs[1], 10, MPI_COMM_WORLD, &reqs[3]);

        MPI_Isend(&(my_C[0][subgridlen]), 1, mpi_column, nbrs[3], 23, MPI_COMM_WORLD, &reqs[4]);
        MPI_Isend(&(my_C[0][1]), 1, mpi_column, nbrs[2], 32, MPI_COMM_WORLD, &reqs[5]);
        MPI_Isend(&(my_C[1][0]), subgridlen+2, MPI_DOUBLE, nbrs[0], 10, MPI_COMM_WORLD, &reqs[6]);
        MPI_Isend(&(my_C[subgridlen][0]), subgridlen+2, MPI_DOUBLE, nbrs[1], 01, MPI_COMM_WORLD, &reqs[7]);

        MPI_Waitall(8, reqs, status);
    }

    return;
}

void update(dmatrix& C, dmatrix& C_old, int N, double h, double u, double v){
    int l,r,a,b;
    double temp;
    for (int i=0;i<N;i++){
        for (int j=0;j<N;j++){
            // this is all to ensure periodic boundary conditions
            r = i+1;
            l = i-1;
            a = j+1;
            b = j-1;
            if (i==0){
                l = N-1;
            } else if (i==N-1){
                r = 0;
            }
            if (j==0){
                b = N-1;
            } else if (j==N-1){
                a = 0;
            }
            temp = 0.25*(C_old[l][j] + C_old[r][j] + C_old[i][b] + C_old[i][a]);
            C[i][j] = temp - h*(u*(C_old[r][j]-C_old[l][j]) + v*(C_old[i][a] - C_old[i][b]));
        }
    }
    return;
}

// with mpi, ghost cells included so don't need to do anything about PBC
// just go from 1 to n instead
void update_mpi(double **C, double **C_old, int n, double h, double u, double v){
    double temp;
    for (int i=1;i<=n;i++){
        for (int j=1;j<=n;j++){
            temp = 0.25*(C_old[i-1][j] + C_old[i+1][j] + C_old[i][j-1] + C_old[i][j+1]);
            C[i][j] = temp - h*(u*(C_old[i+1][j]-C_old[i-1][j]) + v*(C_old[i][j+1] - C_old[i][j-1]));
        }
    }
    return;
}

// initialize matrix 
void init(int N, double dx, dmatrix& C){
    C.resize(N, std::vector<double>(N));
    double x0, y0, sigx2, sigy2;
    double x,y;
    x0 = dx*N/2;
    y0 = x0;
    sigx2 = 0.25*0.25;
    sigy2 = 0.25*0.25;
    for (int i=0; i<N; i++){
        x = (dx*(i+0.5))-x0;
        for (int j=0; j<N; j++){
            y = (dx*(j+0.5))-y0;
            C[i][j] = exp(-(x*x/(2*sigx2) + y*y/(2*sigy2)));
        }
    }
}

void init_mpi(int N, double dx, double **C, int sgl, int* coords){
    // sgl == length of each procs grid
    // make C sgl+2 x sgl+2 for ghost cells on all 4 sides, so shift
    // C entries to (1,sgl+1)
    int xa,ya;
    double x0, y0, sigx2, sigy2;
    double x,y;
    xa = (sgl)*coords[0];
    ya = (sgl)*coords[1];
    x0 = dx*N/2;
    y0 = x0;
    sigx2 = 0.25*0.25;
    sigy2 = 0.25*0.25;
    for (int i=0; i<sgl; i++){
        x = (dx*(xa+i+0.5))-x0;
        for (int j=0; j<sgl; j++){
            y = (dx*(ya+j+0.5))-y0;
            C[i+1][j+1] = exp(-(x*x/(2*sigx2) + y*y/(2*sigy2)));
            //C[i+1][j+1] = (i+1)*(-1*j+1);
        }
    }
}

double **alloc2darray(int size) {
    double *data = (double *)malloc(size*size*sizeof(double));
    double **array = (double **)malloc(size*sizeof(double*));
    for (int i=0;i<size;i++){
        array[i] = &(data[size*i]);
        for (int j=0;j<size;j++){
            array[i][j] = 0.0;
        }
    }
    return array;
}

// copy values from matrix A to B
void copyAtoB(dmatrix& A, dmatrix& B){
    int nx = A.size();
    for (int i=0;i<nx;i++){
        for (int j=0;j<nx;j++){
            B[i][j] = A[i][j];
        }
    }
    return;
}

// print out matrix values
void printMat(int N, dmatrix& C){
    for (int i=0; i<N; i++){
        for (int j=0; j<N; j++){
            printf("%f ",C[i][j]);
        }
        printf("\n");
    }
}

void printToFile(int N, int step, double **A){
    FILE* file;
    char outname [40];
    int n = sprintf(outname, "out-advec-%d.dat", step);
    file = fopen(outname, "w");
    for (int i=0; i<N; i++){
        for (int j=0; j<N; j++){
            fprintf(file, "%f ", A[i][j]);
        }
        fprintf(file, "\n");
    }
    fclose(file);
    return;
}

void printToFile_mpi(int step, double **C, int N, int subgridlen, int ngrid, int myrank){
    // proc 0 is "master" to collect and print all info
    if (myrank == 0){
        MPI_Status status;
        double recvbuf[subgridlen];
        FILE* file;
        char outname [40];
        int n = sprintf(outname, "out-advec-%d.dat", step);
        file = fopen(outname, "w");

        int startp, endp, stat;
        // loop over all N rows (row == row# of whole C matrix)
        // startp is the first proc containing data in that row
        // NOTE: ghost cells included in C, so add 1 to indices
        for (int row=0; row<N; row++){
            startp = (row/subgridlen)*ngrid;
            endp = startp + ngrid;
            printf("row %d of %d startp %d endp %d\n",row,N,startp,endp);

            // if startp = 0, print row before receiving from other procs
            // add 1 to startp then? maybe change to send to self also?
            if (startp==myrank){
                for (int j=1; j<subgridlen+1; j++){
                    fprintf(file, "%f ", C[row+1][j]);
                }
                startp += 1;
            }

            // now loop over processors from startp to startp+ngrid to get partial row info
            for (int i=startp; i<endp; i++){
                stat = MPI_Recv(&recvbuf, subgridlen, MPI_DOUBLE, i, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
                assert(stat == MPI_SUCCESS);
                for (int j=0; j<subgridlen; j++){
                    fprintf(file, "%f ", recvbuf[j]);
                }
            }
            fprintf(file, "\n");
        }
        fclose(file);
    } else {
        // have each other proc send its array one row at a time
        for (int row=0; row<subgridlen; row++){
            MPI_Send(&C[row+1][1], subgridlen, MPI_DOUBLE, 0, 99, MPI_COMM_WORLD);
        }
    }
    return;
}

