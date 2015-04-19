////////////////////////////////////////////////////////////////////////////////
// includes & defines

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <ctime>

#include "device_launch_parameters.h"

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// OpenGL Graphics includes
#include <GL/glew.h>
#if defined (__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#ifndef glutCloseFunc
#define glutCloseFunc glutWMCloseFunc
#endif
#else
#include <GL/freeglut.h>
#endif

// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <timer.h>               // timing functions

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <helper_cuda_gl.h>      // helper functions for CUDA/GL interop

#include <vector_types.h>

#define MAX_EPSILON_ERROR	10.0f
#define THRESHOLD		0.30f
#define REFRESH_DELAY	10 //ms
#define EPS		0.0001f

#ifndef M_PI
#define M_PI	3.14159265358979323846	/* pi */
#endif

#define HEADING(v) (atan2(v.y, v.x))
#define MAX(a,b) ((a > b) ? a : b)

#define CUDA_ERROR_CHECK
#define CudaCheckError() __cudaCheckError( __FILE__, __LINE__ )
inline void __cudaCheckError(const char *file, const int line)
{
#ifdef CUDA_ERROR_CHECK
	cudaError err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n", file, line, cudaGetErrorString(err));
		DEVICE_RESET
		// Make sure we call CUDA Device Reset before exiting
		exit(EXIT_FAILURE);
	}
#endif
	return;
}

////////////////////////////////////////////////////////////////////////////////
// constants

#define RUN_CUDA false

const char *sProgramName = "CUDA fish school";

#define FISH_COUNT (16384)
// 16384 // 8192 // 4096
#define H_SEGMENTS (128)
#define V_SEGMENTS (64)
#define VERTS_IN_FISH (3)

unsigned int window_width = 900;
unsigned int window_height = 600;

struct constants {
	float width;
	float height;
	float scale;

	float peakHeight;
	float trunkWidth;

	float seg_width;
	float seg_height;

	float maxforce;
	float maxspeed;

	float avoidanceRange;
	float alignmentRange;
	float attractionRange;
	float predatorRange;
	float avoidWeight;
	float alignWeight;
	float attractWeight;
	float predatorWeight;
} h_consts;
void set_consts() {
	h_consts.width = 36.0f;
	h_consts.height = 24.0f;
	h_consts.scale = 10.0f;

	h_consts.peakHeight = 0.01f * h_consts.scale;
	h_consts.trunkWidth = 0.002f * h_consts.scale;

	h_consts.seg_width = h_consts.width / (float)H_SEGMENTS;
	h_consts.seg_height = h_consts.height / (float)V_SEGMENTS;

	h_consts.maxforce = 0.06f;
	h_consts.maxspeed = 6.0f;

	h_consts.avoidanceRange = 0.02f * h_consts.scale;
	h_consts.alignmentRange = 0.03f * h_consts.scale;
	h_consts.attractionRange = 0.06f * h_consts.scale;
	h_consts.predatorRange = 0.1f * h_consts.scale;
	h_consts.avoidWeight = 5.5f;
	h_consts.alignWeight = 1.0f;
	h_consts.attractWeight = 1.0f;
	h_consts.predatorWeight = 0.1f;
}
__constant__ constants d_consts;

////////////////////////////////////////////////////////////////////////////////
// vector functions

float2 randomUnit() { float angle = (float) (2 * M_PI * (rand() / (float) RAND_MAX)); return make_float2(cosf(angle), sinf(angle)); }
float2 randomSquare() { return make_float2(rand() / (float) RAND_MAX, rand() / (float) RAND_MAX); }

__host__ __device__ float2 operator+(float2 a, float2 b) { return make_float2(a.x + b.x, a.y + b.y); }
__host__ __device__ float2& operator+=(float2& lhs, const float2& rhs) { lhs.x += rhs.x; lhs.y += rhs.y; return lhs; }
__host__ __device__ float2& operator-=(float2& lhs, const float2& rhs) { lhs.x -= rhs.x; lhs.y -= rhs.y; return lhs; }
__host__ __device__ float2& operator*=(float2& lhs, float rhs) { lhs.x *= rhs; lhs.y *= rhs; return lhs; }
__host__ __device__ float2& operator/=(float2& lhs, float rhs) { lhs.x /= rhs; lhs.y /= rhs; return lhs; }
__host__ __device__ inline float2& v_add(float2& lhs, float2& rhs) { lhs += rhs; return lhs; }
__host__ __device__ inline float2& v_sub(float2& lhs, float2& rhs) { lhs -= rhs; return lhs; }
__host__ __device__ inline float2& v_mul(float2& lhs, float rhs) { return lhs *= rhs; }
__host__ __device__ inline float2& v_div(float2& lhs, float rhs) { return lhs /= rhs; }
__host__ __device__ inline float v_mag(float2& lhs) { return sqrtf(lhs.x*lhs.x + lhs.y*lhs.y); }
__host__ __device__ inline float2& v_norm(float2& lhs) { float m = v_mag(lhs); lhs.x /= m; lhs.y /= m; return lhs; }
__host__ __device__ inline float2& v_setMag(float2& lhs, float m) { v_norm(lhs); lhs.x *= m; lhs.y *= m; return lhs; }
__host__ __device__ inline float2& v_limit(float2& lhs, float max) { if (v_mag(lhs) > max) v_setMag(lhs, max); return lhs; }
__host__ __device__ inline float v_heading(float2& lhs) { return atan2f(lhs.y, lhs.x); }
__host__ __device__ inline float2 v_fromSub(float2& lhs, float2& rhs) { float2 ret(lhs); ret -= rhs; return ret; }
__host__ __device__ inline float v_dist(float2& a, float2& b) { return sqrtf((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y)); }
__host__ __device__ inline float v_fi(float2& a, float2& b) { return atan2f((b.y - a.y), (b.x - a.x)); }
__host__ __device__ float2 operator*(float2 lhs, float rhs) { return lhs *= rhs; }
__host__ __device__ float2 operator/(float2 lhs, float rhs) { return lhs /= rhs; }

////////////////////////////////////////////////////////////////////////////////
// variables

float2 shark = { h_consts.width / 2.0f, h_consts.height / 2.0f };

float2 fish_loc[FISH_COUNT];
float2 fish_vel[FISH_COUNT];
float2 fish_accel[FISH_COUNT];

int fish_seg[FISH_COUNT];
int seg_start[H_SEGMENTS * V_SEGMENTS];
int seg_count[H_SEGMENTS * V_SEGMENTS];

float2 *d_fish_loc = NULL;
float2 *d_fish_vel = NULL;
float2 *d_fish_accel = NULL;
float2 *d_shark = NULL;

int *d_seg = NULL;
int *d_seg_start = NULL;
int *d_seg_count = NULL;

// vbo variables
GLuint vbo;
struct cudaGraphicsResource *cuda_vbo_resource;
void *d_vbo_buffer = NULL;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;

StopWatchInterface *timer = NULL;

// Auto-Verification Code
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
float avgFPS = 0.0f;
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;

int *pArgc = NULL;
char **pArgv = NULL;

cudaStream_t stream[4];

////////////////////////////////////////////////////////////////////////////////
// forward declarations

bool run(int argc, char **argv);
void cleanup();

// GL functionality
bool initGL(int *argc, char **argv);
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res, unsigned int vbo_res_flags);
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res);

// rendering callbacks
void display();
void reshape(int width, int height);
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void passivemotion(int x, int y);
void timerEvent(int value);

// simulation
void runCuda();
void runCPU();

void drawCircle(float cx, float cy, float r, int num_segments);
void randFish(int i);
void sortFish(constants &consts);

///////////////////////////////////////////////////////////////////////////////
//! Kernel functions
///////////////////////////////////////////////////////////////////////////////
#define VERT_ARRAY_POINT(array, index, x, y, z) ( \
	array[4 * (index)] = x, \
	array[4 * (index) + 1] = y, \
	array[4 * (index) + 2] = z, \
	array[4 * (index) + 3] = 1.0f \
)

#define VERT_ARRAY_TRIANGLE(array, index, x1, y1, z1, x2, y2, z2, x3, y3, z3) ( \
	VERT_ARRAY_POINT(array, 3 * (index), x1, y1, z1), \
	VERT_ARRAY_POINT(array, 3 * (index) + 1, x2, y2, z2), \
	VERT_ARRAY_POINT(array, 3 * (index) + 2, x3, y3, z3) \
)

#define ROTATE_POINT_X(x, y, cx, cy, angle) ((x - cx) * cosf(angle) - (y - cy) * sinf(angle) + cx)
#define ROTATE_POINT_Y(x, y, cx, cy, angle) ((x - cx) * sinf(angle) + (y - cy) * cosf(angle) + cy)

__host__ __device__ inline int getVSeg(float y, constants &consts) { return MAX(MIN((int)(y / consts.seg_height), V_SEGMENTS - 1), 0); }
__host__ __device__ inline int getHSeg(float x, constants &consts) { return MAX(MIN((int)(x / consts.seg_width), H_SEGMENTS - 1), 0); }
__host__ __device__ inline int getSeg(float2 &location, constants &consts) { return getVSeg(location.y, consts) * H_SEGMENTS + getHSeg(location.x, consts); }
__host__ __device__ inline int getIndexFrom(int h, int v, int *seg_start) { return seg_start[v * H_SEGMENTS + h]; }
__host__ __device__ inline int getIndexTo(int h, int v, int *seg_start, int *seg_count) { int seg = v * H_SEGMENTS + h; return seg_start[seg] + seg_count[seg]; }

__host__ __device__ void draw_arraw(GLfloat *array, int index, float u, float w, float v, float heading, constants &consts)
{
	VERT_ARRAY_TRIANGLE(array, index,
		ROTATE_POINT_X(u, w - consts.trunkWidth, u, w, heading),
		ROTATE_POINT_Y(u, w - consts.trunkWidth, u, w, heading),
		v,
		ROTATE_POINT_X(u + consts.peakHeight, w, u, w, heading),
		ROTATE_POINT_Y(u + consts.peakHeight, w, u, w, heading),
		v,
		ROTATE_POINT_X(u, w + consts.trunkWidth, u, w, heading),
		ROTATE_POINT_Y(u, w + consts.trunkWidth, u, w, heading),
		v
	);
}

__host__ __device__ float2 seek_func(float2 &location, float2 &velocity, float2 target, constants &consts)
{
	// A vector pointing from the location to the target
	float2 desired(v_fromSub(target, location));
	v_setMag(desired, consts.maxspeed);

	float2 steer = v_fromSub(desired, velocity);
	v_limit(steer, consts.maxforce);
	return steer;
}

__host__ __device__ float2 avoid_func(float2 &location, float2 &velocity, float2 *fish_loc, int *fish_seg, int *seg_start, int *seg_count, constants &consts) {
#define F_RANGE consts.avoidanceRange
	float2 steer = make_float2(0.0f, 0.0f);
	int count = 0;
	for (int v = getVSeg(location.y - F_RANGE, consts); v <= getVSeg(location.y + F_RANGE, consts); ++v) {
		for (int j = getIndexFrom(getHSeg(location.x - F_RANGE, consts), v, seg_start); j < getIndexTo(getHSeg(location.x + F_RANGE, consts), v, seg_start, seg_count); ++j) {
			int i = fish_seg[j];
			float d = v_dist(location, fish_loc[i]);
			if ((d > 0) && (d < F_RANGE)) {
				// Calculate vector pointing away from neighbor
				float2 diff(v_fromSub(location, fish_loc[i]));
				v_norm(diff);
				v_div(diff, d);		// Weight by distance
				v_add(steer, diff);
				count++;
			}
		}
	}
	// Average
	if (count > 0) {
		v_div(steer, (float)count);
	}

	if (v_mag(steer) > 0) {
		v_setMag(steer, consts.maxspeed);
		v_sub(steer, velocity);
		v_limit(steer, consts.maxforce);
	}
	return steer;
#undef F_RANGE
}

// Align == avarage velocity
__host__ __device__ float2 align_func(float2 &location, float2 &velocity, float2 *fish_loc, float2 *fish_vel, int *fish_seg, int *seg_start, int *seg_count, constants &consts) {
#define F_RANGE consts.alignmentRange
	float2 sum = { 0.0f, 0.0f };
	int count = 0;
	for (int v = getVSeg(location.y - F_RANGE, consts); v <= getVSeg(location.y + F_RANGE, consts); ++v) {
		for (int j = getIndexFrom(getHSeg(location.x - F_RANGE, consts), v, seg_start); j < getIndexTo(getHSeg(location.x + F_RANGE, consts), v, seg_start, seg_count); ++j) {
			int i = fish_seg[j];
			float d = v_dist(location, fish_loc[i]);
			if ((d > 0) && (d < F_RANGE)) {
				v_add(sum, fish_vel[i]);
				count++;
			}
		}
	}
	if (count > 0) {
		v_div(sum, (float)count);
		v_setMag(sum, consts.maxspeed);
		float2 steer = v_fromSub(sum, velocity);
		v_limit(steer, consts.maxforce);
		return steer;
	}
	return make_float2(0.0f, 0.0f);
#undef F_RANGE
}

// Attraction == avarage location
__host__ __device__ float2 attract_func(float2 &location, float2 &velocity, float2 *fish_loc, int *fish_seg, int *seg_start, int *seg_count, constants &consts) {
#define F_RANGE consts.attractionRange
	float2 sum = { 0.0f, 0.0f };
	int count = 0;
	for (int v = getVSeg(location.y - F_RANGE, consts); v <= getVSeg(location.y + F_RANGE, consts); ++v) {
		for (int j = getIndexFrom(getHSeg(location.x - F_RANGE, consts), v, seg_start); j < getIndexTo(getHSeg(location.x + F_RANGE, consts), v, seg_start, seg_count); ++j) {
			int i = fish_seg[j];
			float d = v_dist(location, fish_loc[i]);
			if ((d > 0) && (d < F_RANGE)) {
				v_add(sum, fish_loc[i]);
				count++;
			}
		}
	}
	if (count > 0) {
		v_div(sum, (float)count);
		return seek_func(location, velocity, sum, consts);
	}
	else {
		return make_float2(0.0f, 0.0f);
	}
#undef F_RANGE
}

__host__ __device__ float2 predator_func(float2 &location, float2 &shark, constants &consts) {
#define F_RANGE consts.predatorRange
	float d = v_dist(location, shark);
	if (d > 0 && d < F_RANGE) {
		float2 run = v_fromSub(shark, location);
		float h = (float)MIN(HEADING(run) + M_PI / 4.0f, HEADING(run) - M_PI / 4.0f);
		run = make_float2(cosf(h), sinf(h));
		run *= -1;
		v_norm(run);
		v_div(run, d);
		return run;
	}
	return make_float2(0.0f, 0.0f);
#undef F_RANGE
}

__host__ __device__ void border_wrap_func(float2 &location, constants &consts)
{
	if (location.x < 0.0f) location.x = consts.width;
	if (location.y < 0.0f) location.y = consts.height;
	if (location.x > consts.width) location.x = 0.0f;
	if (location.y > consts.height) location.y = 0.0f;
}

__host__ __device__ void update_func(float2 &location, float2 &velocity, float2 &acceleration, float dt, constants &consts)
{
	velocity += acceleration;
	v_limit(velocity, consts.maxspeed);
	location += velocity * dt;
	acceleration *= 0;
	border_wrap_func(location, consts);
}

__global__ void avoid_kernel(float2 *location, float2 *velocity, float2 *acceleration, int *seg, int *seg_start, int *seg_count)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i > FISH_COUNT) return;
	acceleration[i] += avoid_func(location[i], velocity[i], location, seg, seg_start, seg_count, d_consts) * d_consts.avoidWeight;
}

__global__ void align_kernel(float2 *location, float2 *velocity, float2 *acceleration, int *seg, int *seg_start, int *seg_count)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i > FISH_COUNT) return;
	acceleration[i] += align_func(location[i], velocity[i], location, velocity, seg, seg_start, seg_count, d_consts) * d_consts.alignWeight;
}

__global__ void attract_kernel(float2 *location, float2 *velocity, float2 *acceleration, int *seg, int *seg_start, int *seg_count)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i > FISH_COUNT) return;
	acceleration[i] += attract_func(location[i], velocity[i], location, seg, seg_start, seg_count, d_consts) * d_consts.attractWeight;
}

__global__ void predator_kernel(float2 *location, float2 *acceleration, float2 *shark)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i > FISH_COUNT) return;
	acceleration[i] += predator_func(location[i], *shark, d_consts) * d_consts.predatorWeight;
}

__global__ void update_draw_kernel(GLfloat *vertices, float2 *location, float2 *velocity, float2 *acceleration, float dt)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i > FISH_COUNT) return;
	update_func(location[i], velocity[i], acceleration[i], dt, d_consts);
	draw_arraw(vertices, i, location[i].x, location[i].y, 1.0f, HEADING(velocity[i]), d_consts);
}

void launch_kernel(GLfloat *fish_verts, float2 *fish_loc, float2 *fish_vel, float2 *fish_accel, int *seg, int *seg_start, int *seg_count, float2 *shark, float dt)
{
    // execute the kernel
	int blockSize = 256;
	dim3 threadsInBlock(blockSize);
	dim3 blocksInGrid((int)(FISH_COUNT / (float)blockSize));
	avoid_kernel << < blocksInGrid, threadsInBlock, 0, stream[0] >> >(fish_loc, fish_vel, fish_accel, seg, seg_start, seg_count);
	CudaCheckError();
	align_kernel << < blocksInGrid, threadsInBlock, 0, stream[1] >> >(fish_loc, fish_vel, fish_accel, seg, seg_start, seg_count);
	CudaCheckError();
	attract_kernel << < blocksInGrid, threadsInBlock, 0, stream[2] >> >(fish_loc, fish_vel, fish_accel, seg, seg_start, seg_count);
	CudaCheckError();
	predator_kernel << < blocksInGrid, threadsInBlock, 0, stream[3] >> >(fish_loc, fish_accel, shark);
	CudaCheckError();
	checkCudaErrors(cudaDeviceSynchronize());
	update_draw_kernel << < blocksInGrid, threadsInBlock >> >(fish_verts, fish_loc, fish_vel, fish_accel, dt);
	CudaCheckError();
}

bool checkHW(char *name, const char *gpuType, int dev)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    strcpy(name, deviceProp.name);

    if (!STRNCASECMP(deviceProp.name, gpuType, strlen(gpuType))) {
        return true;
    }
    else {
        return false;
    }
}

//int findGraphicsGPU(char *name)
//{
//    int nGraphicsGPU = 0;
//    int deviceCount = 0;
//    bool bFoundGraphics = false;
//    char firstGraphicsName[256], temp[256];
//
//    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
//
//    if (error_id != cudaSuccess)
//    {
//        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
//        printf("> FAILED %s sample finished, exiting...\n", sProgramName);
//        exit(EXIT_FAILURE);
//    }
//
//    // This function call returns 0 if there are no CUDA capable devices.
//    if (deviceCount == 0)
//    {
//        printf("> There are no device(s) supporting CUDA\n");
//        return false;
//    }
//    else
//    {
//        printf("> Found %d CUDA Capable Device(s)\n", deviceCount);
//    }
//
//    for (int dev = 0; dev < deviceCount; ++dev)
//    {
//        bool bGraphics = !checkHW(temp, (const char *)"Tesla", dev);
//        printf("> %s\t\tGPU %d: %s\n", (bGraphics ? "Graphics" : "Compute"), dev, temp);
//
//        if (bGraphics)
//        {
//            if (!bFoundGraphics)
//            {
//                strcpy(firstGraphicsName, temp);
//            }
//
//            nGraphicsGPU++;
//        }
//    }
//
//    if (nGraphicsGPU)
//    {
//        strcpy(name, firstGraphicsName);
//    }
//    else
//    {
//        strcpy(name, "this hardware");
//    }
//
//    return nGraphicsGPU;
//}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
	pArgc = &argc;
	pArgv = argv;

#if defined(__linux__)
	setenv("DISPLAY", ":0", 0);
#endif

	printf("%s starting...\n", sProgramName);

	printf("\n");

	run(argc, argv);

	printf("%s completed, returned %s\n", sProgramName, (g_TotalErrors == 0) ? "OK" : "ERROR!");
	exit(g_TotalErrors == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
}

void computeFPS()
{
	frameCount++;
	fpsCount++;

	if (fpsCount == fpsLimit)
	{
		avgFPS = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
		fpsCount = 0;
		fpsLimit = (int)MAX(avgFPS, 1.f);

		sdkResetTimer(&timer);
	}

	char fps[256];
	sprintf(fps, "Cuda GL Interop (VBO): %3.1f fps (Max 100Hz)", avgFPS);
	glutSetWindowTitle(fps);
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
bool initGL(int *argc, char **argv)
{
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(window_width, window_height);
	glutCreateWindow("Cuda GL Interop (VBO)");
	glutDisplayFunc(display);
	glutReshapeFunc(reshape);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutPassiveMotionFunc(passivemotion);
	glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
#if defined (__APPLE__) || defined(MACOSX)
	atexit(cleanup);
#else
	glutCloseFunc(cleanup);
#endif

	// initialize necessary OpenGL extensions
	glewInit();

	if (!glewIsSupported("GL_VERSION_2_0 "))
	{
		fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
		fflush(stderr);
		return false;
	}

	// default initialization
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glDisable(GL_DEPTH_TEST);

	// viewport
	glViewport(0, 0, window_width, window_height);

	// projection
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	//gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.1, 10.0);
	glOrtho(0.0, h_consts.width, 0.0, h_consts.height, 0.1, 10.0);

	SDK_CHECK_ERROR_GL();

	return true;
}


////////////////////////////////////////////////////////////////////////////////
//! Run
////////////////////////////////////////////////////////////////////////////////
bool run(int argc, char **argv/*, char *ref_file*/)
{
	set_consts();

	// Create the CUTIL timer
	sdkCreateTimer(&timer);

	// First initialize OpenGL context, so we can properly set the GL for CUDA.
	// This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
	if (false == initGL(&argc, argv))
	{
		return false;
	}

	if (RUN_CUDA) {
		// use command-line specified CUDA device, otherwise use device with highest Gflops/s
		if (checkCmdLineFlag(argc, (const char **)argv, "device"))
		{
			if (gpuGLDeviceInit(argc, (const char **)argv) == -1)
			{
				return false;
			}
		}
		else
		{
			cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());
		}
	}

	// create VBO
	createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);

	srand((unsigned int)time(NULL));

	for (int i = 0; i < FISH_COUNT; ++i) {
		randFish(i);
	}
	sortFish(h_consts);

	if (RUN_CUDA) {
		for (int i = 0; i < 4; ++i)
			checkCudaErrors(cudaStreamCreate(&stream[i]));

		// copy fish & shark to device memory
		int size = FISH_COUNT * sizeof(float2);
		checkCudaErrors(cudaMalloc((void **) &d_fish_loc, size));
		checkCudaErrors(cudaMemcpy(d_fish_loc, &fish_loc, size, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMalloc((void **) &d_fish_vel, size));
		checkCudaErrors(cudaMemcpy(d_fish_vel, &fish_vel, size, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMalloc((void **) &d_fish_accel, size));
		checkCudaErrors(cudaMemcpy(d_fish_accel, &fish_accel, size, cudaMemcpyHostToDevice));

		size = FISH_COUNT * sizeof(int);
		checkCudaErrors(cudaMalloc((void **) &d_seg, size));
		checkCudaErrors(cudaMemcpy(d_seg, &fish_seg, size, cudaMemcpyHostToDevice));

		size = H_SEGMENTS * V_SEGMENTS * sizeof(int);
		checkCudaErrors(cudaMalloc((void **) &d_seg_start, size));
		checkCudaErrors(cudaMemcpy(d_seg_start, &seg_start, size, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMalloc((void **) &d_seg_count, size));
		checkCudaErrors(cudaMemcpy(d_seg_count, &seg_count, size, cudaMemcpyHostToDevice));

		checkCudaErrors(cudaMalloc((void **)&d_shark, sizeof(float2)));
		checkCudaErrors(cudaMemcpy(d_shark, &shark, sizeof(float2), cudaMemcpyHostToDevice));

		checkCudaErrors(cudaMemcpyToSymbol(d_consts, &h_consts, sizeof(constants)));
	}

	// run the cuda part
	if (RUN_CUDA)
		runCuda();
	else
		runCPU();

	// start rendering mainloop
	glutMainLoop();

	return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runCuda()
{
    // map OpenGL buffer object for writing from CUDA
    GLfloat *dptr;
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_vbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, cuda_vbo_resource));
    //printf("CUDA mapped VBO: May access %ld bytes\n", num_bytes);

	checkCudaErrors(cudaMemcpy(d_shark, &shark, sizeof(float2), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemset(d_fish_accel, 0, FISH_COUNT * sizeof(float2)));

	checkCudaErrors(cudaMemcpy(d_seg, fish_seg, FISH_COUNT * sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_seg_start, seg_start, H_SEGMENTS * V_SEGMENTS * sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_seg_count, seg_count, H_SEGMENTS * V_SEGMENTS * sizeof(int), cudaMemcpyHostToDevice));

	float dt = 0.013f;

	launch_kernel(dptr, d_fish_loc, d_fish_vel, d_fish_accel, d_seg, d_seg_start, d_seg_count, d_shark, dt);

	checkCudaErrors(cudaMemcpy(fish_loc, d_fish_loc, FISH_COUNT * sizeof(float2), cudaMemcpyDeviceToHost));

	sortFish(h_consts);

    // unmap buffer object
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));
}

void runCPU()
{
	unsigned int size = FISH_COUNT * VERTS_IN_FISH * 4 * sizeof(float);
	GLfloat *fish_verts = (GLfloat*)malloc(size);

	float dt = 0.013f;

	// Compute
	for (int i = 0; i < FISH_COUNT; ++i) {
		fish_accel[i] += avoid_func(fish_loc[i], fish_vel[i], fish_loc, fish_seg, seg_start, seg_count, h_consts) * h_consts.avoidWeight;
		fish_accel[i] += align_func(fish_loc[i], fish_vel[i], fish_loc, fish_vel, fish_seg, seg_start, seg_count, h_consts) * h_consts.alignWeight;
		fish_accel[i] += attract_func(fish_loc[i], fish_vel[i], fish_loc, fish_seg, seg_start, seg_count, h_consts) * h_consts.attractWeight;
		fish_accel[i] += predator_func(fish_loc[i], shark, h_consts) * h_consts.predatorWeight;
	}

	for (int i = 0; i < FISH_COUNT; ++i) {
		update_func(fish_loc[i], fish_vel[i], fish_accel[i], dt, h_consts);
		draw_arraw(fish_verts, i, fish_loc[i].x, fish_loc[i].y, 1.0f, HEADING(fish_vel[i]), h_consts);
	}

	sortFish(h_consts);

	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, size, fish_verts, GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	free(fish_verts);
}

////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res, unsigned int vbo_res_flags)
{
	assert(vbo);

	// create buffer object
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);

	// initialize buffer object
	unsigned int size = FISH_COUNT * VERTS_IN_FISH * 4 * sizeof(float);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	if (RUN_CUDA) {
		// register this buffer object with CUDA
		checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));
	}

	SDK_CHECK_ERROR_GL();
}

////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res)
{
	if (RUN_CUDA) {
		// unregister this buffer object with CUDA
		checkCudaErrors(cudaGraphicsUnregisterResource(vbo_res));
	}

	glBindBuffer(1, *vbo);
	glDeleteBuffers(1, vbo);

	*vbo = 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display()
{
	sdkStartTimer(&timer);

	// run CUDA kernel to generate vertex positions
	if (RUN_CUDA)
		runCuda();
	else
		runCPU();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glColor3f(0.0f, 0.0f, 1.0f);
	glBegin(GL_LINES);
	for (int i = 1; i < H_SEGMENTS; ++i) {
		glVertex2f(i * h_consts.width / (float)H_SEGMENTS, 0.0f);
		glVertex2f(i * h_consts.width / (float)H_SEGMENTS, h_consts.height);
	}
	for (int i = 1; i < V_SEGMENTS; ++i) {
		glVertex2f(0.0f, i * h_consts.height / (float)V_SEGMENTS);
		glVertex2f(h_consts.width, i * h_consts.height / (float)V_SEGMENTS);
	}
	glEnd();

	glColor3f(0.0f, 1.0f, 0.0f);
	drawCircle(fish_loc[0].x, fish_loc[0].y, h_consts.avoidanceRange, 30);
	drawCircle(fish_loc[0].x, fish_loc[0].y, h_consts.alignmentRange, 30);
	drawCircle(fish_loc[0].x, fish_loc[0].y, h_consts.attractionRange, 30);

	// set view matrix
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.0, 0.0, translate_z);
//	glRotatef(rotate_x, 1.0, 0.0, 0.0);
//	glRotatef(rotate_y, 0.0, 1.0, 0.0);

	// render from the vbo
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glVertexPointer(4, GL_FLOAT, 0, 0);

	glEnableClientState(GL_VERTEX_ARRAY);
	glColor3f(1.0, 0.0, 0.0);
	glDrawArrays(GL_TRIANGLES, 0, FISH_COUNT * VERTS_IN_FISH);
	glDisableClientState(GL_VERTEX_ARRAY);

	glColor3f(0.0f, 1.0f, 0.0f);
	drawCircle(shark.x, shark.y, h_consts.predatorRange, 30);

	glutSwapBuffers();

	sdkStopTimer(&timer);
	computeFPS();
}

void reshape(int width, int height)
{
	window_width = width;
	window_height = height;
	glViewport(0, 0, window_width, window_height);
}

void timerEvent(int value)
{
	if (glutGetWindow())
	{
		glutPostRedisplay();
		glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
	}
}

void cleanup()
{
	sdkDeleteTimer(&timer);

	if (vbo)
	{
		deleteVBO(&vbo, cuda_vbo_resource);
	}

	if (RUN_CUDA) {
		for (int i = 0; i < 4; ++i)
			checkCudaErrors(cudaStreamDestroy(stream[i]));

		checkCudaErrors(cudaFree(d_fish_loc));
		checkCudaErrors(cudaFree(d_fish_vel));
		checkCudaErrors(cudaFree(d_fish_accel));
		checkCudaErrors(cudaFree(d_seg));
		checkCudaErrors(cudaFree(d_seg_start));
		checkCudaErrors(cudaFree(d_seg_count));
		checkCudaErrors(cudaFree(d_shark));

		// cudaDeviceReset causes the driver to clean up all state. While
		// not mandatory in normal operation, it is good practice.  It is also
		// needed to ensure correct operation when the application is being
		// profiled. Calling cudaDeviceReset causes all profile data to be
		// flushed before the application exits
		cudaDeviceReset();
	}
}


////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
	switch (key)
	{
	case (27) :
#if defined(__APPLE__) || defined(MACOSX)
		exit(EXIT_SUCCESS);
#else
		glutDestroyWindow(glutGetWindow());
		return;
#endif
	}
}

////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
void mouse(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
	{
		mouse_buttons |= 1 << button;
	}
	else if (state == GLUT_UP)
	{
		mouse_buttons = 0;
	}

	mouse_old_x = x;
	mouse_old_y = y;
}

void motion(int x, int y)
{
	float dx, dy;
	dx = (float)(x - mouse_old_x);
	dy = (float)(y - mouse_old_y);

	if (mouse_buttons & 1)
	{
		rotate_x += dy * 0.2f;
		rotate_y += dx * 0.2f;
	}
	else if (mouse_buttons & 4)
	{
		translate_z += dy * 0.01f;
	}

	mouse_old_x = x;
	mouse_old_y = y;
}

void passivemotion(int x, int y)
{
	shark.x = h_consts.width * x / (float)window_width;
	shark.y = h_consts.height - h_consts.height * y / (float)window_height;
}

inline void drawCircle(float cx, float cy, float r, int num_segments)
{
	float theta = (float) (2 * M_PI / (float)num_segments);
	float c = cosf(theta); // precalculate the sine and cosine
	float s = sinf(theta);
	float t;

	float x = r; //we start at angle = 0
	float y = 0;

	glBegin(GL_LINE_LOOP);
	for (int ii = 0; ii < num_segments; ii++)
	{
		glVertex2f(x + cx, y + cy); //output vertex
		//apply the rotation matrix
		t = x;
		x = c * x - s * y;
		y = s * t + c * y;
	}
	glEnd();
}

void randFish(int i)
{
	float2 random = randomUnit();
	fish_loc[i] = make_float2(h_consts.width * (rand() / (float)RAND_MAX), h_consts.height * (rand() / (float)RAND_MAX));
//	fish_loc[i] = make_float2(0.5f * width, 0.5f * height);
	random = randomUnit();
	v_setMag(random, h_consts.maxspeed);
	fish_vel[i] = make_float2(random.x, random.y);
}

void sortFish(constants &consts)
{
	// Init
	int seg_last[H_SEGMENTS * V_SEGMENTS];
	for (int i = 0; i < H_SEGMENTS * V_SEGMENTS; ++i) {
		seg_count[i] = 0;
	}
	// Set each segment fish count
	for (int i = 0; i < FISH_COUNT; ++i) {
		++seg_count[getSeg(fish_loc[i], consts)];
	}
	// Set segment start indexes
	int start = 0;
	for (int i = 0; i < H_SEGMENTS * V_SEGMENTS; ++i) {
		seg_start[i] = start;
		seg_last[i] = start;
		start += seg_count[i];
	}
	// Sort fish by segment in index array
	for (int i = 0; i < FISH_COUNT; ++i) {
		fish_seg[seg_last[getSeg(fish_loc[i], consts)]++] = i;
	}
}
