
#include <omp.h>
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
void init_mpi(int N, double dx, double **C, int subgridlen, int* coords, int nthreads);
void exchangeGhostCells(double **my_C, int subgridlen, MPI_Comm cartcomm, int* nbrs, int* coords, int mype, int blockflag);
void update(double **C, double **C_old, int N, double h, double u, double v);
void update_mpi(double **C, double **C_old, int n, double h, double u, double v, int nthreads);
void printToFile(int N, int step, double **A);
void printToFile_mpi(int step, double **C, int N, int subgridlen, int ngrid, int myrank);

int main(int argc, char *argv[]){
    int rflag;
    rflag = 1;
    // parameters: read in
    double NT, T, L, u, v, dx, dt, h, c0, c1, localtime, globaltime;
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

    dx = L/N;
    dt = T/NT;
    h = dx/sqrt(2*(u*u+v*v));
    if (dt>h){
        printf("UNSTABLE! h: %f \ndt: %f\n",h,dt);
        return 1;
    }
    h = dt/(2*dx);

    // SETUP MPI STUFF

    int nprocs, stat, mype, ngrid, subgridlen;
    MPI_Init(&argc, &argv);
    stat = MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    assert(stat == MPI_SUCCESS);
    stat = MPI_Comm_rank(MPI_COMM_WORLD, &mype);
    assert(stat == MPI_SUCCESS);
    assert(fmod(N, sqrt(nprocs)) == 0);

    if (mpimode.compare("mpi_non_blocking") == 0) {
        blockflag = 0;
        if (mype == 0){
 //           printf("MPI non-blocking mode\n");
            printf("non-blocking ");           
            if (nthreads != 1) {
                printf("Note: reseting to 1 OMP thread\n");
            }
        }
        nthreads = 1;
    } else if (mpimode.compare("mpi_blocking") == 0){
        blockflag = 1;
        if (mype == 0){
 //           printf("MPI blocking mode\n");
            printf("blocking     ");           
            if (nthreads != 1) {
                printf("Note: reseting to 1 OMP thread\n");
            }
        }
        nthreads = 1;
    } else if (mpimode.compare("hybrid") == 0) {
        blockflag = 0;
        if (mype == 0){
 //           printf("hybrid mode with %d mpi ranks and %d threads per rank (non-blocking MPI) \n",nprocs, nthreads);
            printf("hybrid       ");           
            if (nthreads == 1) {
                printf("Note: only 1 OMP thread\n");
            }
        }
    } else {
        printf("non valid MPI mode\n");
        return 1;
    }

    if (mype==0){
 //       printf("N: %d \nNT: %f \nL: %f \nT: %f \nu: %f \nv: %f \n", N, NT, L, T, u, v);
        printf("%d  %d  ", nprocs, nthreads); 
    }

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
    
    // initialize matrix portion on each proc
    // ghost cells included in C, so C is size subgridlen+2 
    // cells "owned" by C are in the interior
    double **my_C;
    double **my_C_old;
    my_C = alloc2darray(subgridlen+2);
    my_C_old = alloc2darray(subgridlen+2);
    init_mpi(N, dx, my_C, subgridlen, coords, nthreads);
    init_mpi(N, dx, my_C_old, subgridlen, coords, nthreads);

   // printToFile(subgridlen+2, mype, my_C);

    // get neighbors: nbrs: (up,down,left,right)
    int nbrs[4];
    stat = MPI_Cart_shift(cartcomm, 0, 1, &nbrs[0], &nbrs[1]);
    stat = MPI_Cart_shift(cartcomm, 1, 1, &nbrs[2], &nbrs[3]);

    c0 = omp_get_wtime();
    // run for NT timesteps
    for (int step=0;step<NT;step++){
        std::swap(my_C,my_C_old);
        if (nprocs == 1) {
            update(my_C, my_C_old, subgridlen, h, u, v);
        } else {
            exchangeGhostCells(my_C_old, subgridlen, cartcomm, nbrs, coords, mype, blockflag);
            update_mpi(my_C, my_C_old, subgridlen, h, u, v, nthreads);
        }
    //    if (step%10 == 0){
    //        printToFile_mpi(step, my_C, N, subgridlen, ngrid, mype);
    //    }
    }
    c1 = omp_get_wtime();
    localtime = c1-c0;
    MPI_Reduce(&localtime, &globaltime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (mype==0) {
        printf(" %f\n", globaltime/nprocs);
    }
  //  printToFile_mpi(NT, my_C, N, subgridlen, ngrid, mype);
    MPI_Barrier(cartcomm);
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
        // blocking send/recv
        if (coords[1] == 0) {
            // SEND RIGHT SIDE TO RIGHT NEGIHBOR 
          // printf("Send right %d to %d\n",mype, nbrs[3]);
            MPI_Send(&(my_C[0][subgridlen]), 1, mpi_column, nbrs[3], 23, MPI_COMM_WORLD);
          // printf("Receive from left %d from %d\n",mype, nbrs[2]);
            stat = MPI_Recv(&(my_C[0][0]), 1, mpi_column, nbrs[2], 23, MPI_COMM_WORLD, &status);
            assert(stat == MPI_SUCCESS);

            // SEND LEFT SIDE TO LEFT NEGIHBOR 
          // printf("Send left %d to %d\n",mype, nbrs[2]);
            MPI_Send(&(my_C[0][1]), 1, mpi_column, nbrs[2], 32, MPI_COMM_WORLD);
          // printf("Receive from right %d from %d\n",mype, nbrs[3]);
            stat = MPI_Recv(&(my_C[0][subgridlen+1]), 1, mpi_column, nbrs[3], 32, MPI_COMM_WORLD, &status);
            assert(stat == MPI_SUCCESS);
        } else {
            // SEND RIGHT SIDE TO RIGHT NEGIHBOR 
          // printf("Receive from left %d from %d\n",mype, nbrs[2]);
            stat = MPI_Recv(&(my_C[0][0]), 1, mpi_column, nbrs[2], 23, MPI_COMM_WORLD, &status);
            assert(stat == MPI_SUCCESS);
          // printf("Send right %d to %d\n",mype, nbrs[3]);
            MPI_Send(&(my_C[0][subgridlen]), 1, mpi_column, nbrs[3], 23, MPI_COMM_WORLD);

            // SEND LEFT SIDE TO LEFT NEGIHBOR 
          // printf("Receive from right %d from %d\n",mype, nbrs[3]);
            stat = MPI_Recv(&(my_C[0][subgridlen+1]), 1, mpi_column, nbrs[3], 32, MPI_COMM_WORLD, &status);
            assert(stat == MPI_SUCCESS);
          // printf("Send left %d to %d\n",mype, nbrs[2]);
            MPI_Send(&(my_C[0][1]), 1, mpi_column, nbrs[2], 32, MPI_COMM_WORLD);
        }
        if (coords[0] == 0){
            // SEND TOP TO ABOVE NEIGHBOR
          // printf("Send top %d to %d\n",mype, nbrs[0]);
            MPI_Send(&(my_C[1][0]), subgridlen+2, MPI_DOUBLE, nbrs[0], 10, MPI_COMM_WORLD);
          // printf("Receive from below %d from %d\n",mype, nbrs[1]);
            stat = MPI_Recv(&(my_C[subgridlen+1][0]), subgridlen+2, MPI_DOUBLE, nbrs[1], 10, MPI_COMM_WORLD, &status);
            assert(stat == MPI_SUCCESS);

            // SEND BOTTOM TO BELOW NEIGHBOR
          // printf("Send bottom %d to %d\n",mype, nbrs[0]);
            MPI_Send(&(my_C[subgridlen][0]), subgridlen+2, MPI_DOUBLE, nbrs[1], 01, MPI_COMM_WORLD);
          // printf("Receive from top %d from %d\n",mype, nbrs[0]);
            stat = MPI_Recv(&(my_C[0][0]), subgridlen+2, MPI_DOUBLE, nbrs[0], 01, MPI_COMM_WORLD, &status);
            assert(stat == MPI_SUCCESS);
        } else {
          // printf("Receive from below %d from %d\n",mype, nbrs[1]);
            stat = MPI_Recv(&(my_C[subgridlen+1][0]), subgridlen+2, MPI_DOUBLE, nbrs[1], 10, MPI_COMM_WORLD, &status);
            assert(stat == MPI_SUCCESS);
          // printf("Send top %d to %d\n",mype, nbrs[0]);
            MPI_Send(&(my_C[1][0]), subgridlen+2, MPI_DOUBLE, nbrs[0], 10, MPI_COMM_WORLD);

          // printf("Receive from top %d from %d\n",mype, nbrs[0]);
            stat = MPI_Recv(&(my_C[0][0]), subgridlen+2, MPI_DOUBLE, nbrs[0], 01, MPI_COMM_WORLD, &status);
            assert(stat == MPI_SUCCESS);
          // printf("Send bottom %d to %d\n",mype, nbrs[0]);
            MPI_Send(&(my_C[subgridlen][0]), subgridlen+2, MPI_DOUBLE, nbrs[1], 01, MPI_COMM_WORLD);
        }
    } else {
        // non-blocking send/recv
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

// using update when nprocs is 1, change to 1...N, not 0...N-1
void update(double **C, double **C_old, int N, double h, double u, double v){
    int l,r,a,b;
    double temp;
    for (int i=1;i<=N;i++){
        for (int j=1;j<=N;j++){
            // this is all to ensure periodic boundary conditions
            r = i+1;
            l = i-1;
            a = j+1;
            b = j-1;
            if (i==1){
                l = N;
            } else if (i==N){
                r = 1;
            }
            if (j==1){
                b = N;
            } else if (j==N){
                a = 1;
            }
            temp = 0.25*(C_old[l][j] + C_old[r][j] + C_old[i][b] + C_old[i][a]);
            C[i][j] = temp - h*(u*(C_old[r][j]-C_old[l][j]) + v*(C_old[i][a] - C_old[i][b]));
        }
    }
    return;
}

// with mpi, ghost cells included so don't need to do anything about PBC
// just go from 1 to n instead
void update_mpi(double **C, double **C_old, int n, double h, double u, double v, int nthreads){
    double temp;
    int i,j;
    if (nthreads > 1) {
#pragma omp parallel for num_threads(nthreads) private(i,j,temp) shared(n,C,C_old) schedule(static)
        for (i=1;i<=n;i++){
            for (j=1;j<=n;j++){
                temp = 0.25*(C_old[i-1][j] + C_old[i+1][j] + C_old[i][j-1] + C_old[i][j+1]);
                C[i][j] = temp - h*(u*(C_old[i+1][j]-C_old[i-1][j]) + v*(C_old[i][j+1] - C_old[i][j-1]));
            }
        }

    } else {
        for (i=1;i<=n;i++){
            for (j=1;j<=n;j++){
                temp = 0.25*(C_old[i-1][j] + C_old[i+1][j] + C_old[i][j-1] + C_old[i][j+1]);
                C[i][j] = temp - h*(u*(C_old[i+1][j]-C_old[i-1][j]) + v*(C_old[i][j+1] - C_old[i][j-1]));
            }
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

void init_mpi(int N, double dx, double **C, int sgl, int* coords, int nthreads){
    // sgl == length of each procs grid
    // make C sgl+2 x sgl+2 for ghost cells on all 4 sides, so shift
    // C entries to (1,sgl+1)
    int xa,ya,i,j;
    double x0, y0, sigx2, sigy2;
    double x,y;
    xa = (sgl)*coords[0];
    ya = (sgl)*coords[1];
    x0 = dx*N/2;
    y0 = x0;
    sigx2 = 0.25*0.25;
    sigy2 = 0.25*0.25;
    if (nthreads > 1) {
#pragma omp parallel for num_threads(nthreads) private(i,j,x,y) shared(C,sgl,x0,y0,sigx2,sigy2,xa,ya) schedule(static)
        for (int i=0; i<sgl; i++){
            x = (dx*(xa+i+0.5))-x0;
            for (int j=0; j<sgl; j++){
                y = (dx*(ya+j+0.5))-y0;
                C[i+1][j+1] = exp(-(x*x/(2*sigx2) + y*y/(2*sigy2)));
                //C[i+1][j+1] = (i+1)*(-1*j+1);
            }
        }
    } else {
        for (int i=0; i<sgl; i++){
            x = (dx*(xa+i+0.5))-x0;
            for (int j=0; j<sgl; j++){
                y = (dx*(ya+j+0.5))-y0;
                C[i+1][j+1] = exp(-(x*x/(2*sigx2) + y*y/(2*sigy2)));
                //C[i+1][j+1] = (i+1)*(-1*j+1);
            }
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
          // printf("row %d of %d startp %d endp %d\n",row,N,startp,endp);

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

