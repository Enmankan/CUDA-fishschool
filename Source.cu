// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

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
#define M_PI	3.14159265358979323846	/* pi */

#define HEADING(v) (atan2(v.y, v.x))

////////////////////////////////////////////////////////////////////////////////
// constants
const unsigned int window_width = 1200;
const unsigned int window_height = 800;

const float width = 12.0f;
const float height = 8.0f;
const float scale = 10.0f;
__device__ const float d_width = 12.0f;
__device__ const float d_height = 8.0f;
__device__ const float d_scale = 10.0f;

const unsigned int VERTS_IN_FISH = 3;
const unsigned int FISH_COUNT = 2048;

const float avoidanceRange = 0.08f * scale;
const float alignmentRange = 0.04f * scale;
const float attractionRange = 0.08f * scale;
const float predatorRange = 0.1f * scale;
__device__ const float d_avoidanceRange = 0.02f * 10.0f;
__device__ const float d_alignmentRange = 0.04f * 10.0f;
__device__ const float d_attractionRange = 0.08f * 10.0f;
__device__ const float d_predatorRange = 0.1f * 10.0f;
//const float max_speed = 0.3f * scale;

//struct Vector2 {
//	float x;
//	float y;
//	Vector2(float x, float y) : x(x), y(y) {}
//	Vector2(Vector2 const& v) : x(v.x), y(v.y) {}
//	Vector2& operator+=(const Vector2& rhs) { x += rhs.x; y += rhs.y; return *this; }
//	Vector2& operator-=(const Vector2& rhs) { x -= rhs.x; y -= rhs.y; return *this; }
//	Vector2& operator*=(float rhs) { x *= rhs; y *= rhs; return *this; }
//	Vector2& operator/=(float rhs) { x /= rhs; y /= rhs; return *this; }
//	static Vector2 randomUnit() { float angle = 2 * M_PI * (rand() % 360) / 360; return Vector2(cos(angle), sin(angle)); }
//	Vector2& limit(float max) { if (mag() > max) setMag(max); return *this; }
//	Vector2& normalize() { float m = mag(); x /= m; y /= m; return *this; }
//	Vector2& setMag(float m) { normalize(); x *= m; y *= m; return *this; }
//	Vector2& add(Vector2& rhs) { *this += rhs; return *this; }
//	Vector2& sub(Vector2& rhs) { *this -= rhs; return *this; }
//	static Vector2 sub(Vector2& lhs, Vector2& rhs) { Vector2 ret(lhs); ret -= rhs; return ret; }
//	Vector2& mult(float rhs) { return *this *= rhs; }
//	Vector2& div(float rhs) { return *this /= rhs; }
//	float mag() { return sqrt(x*x + y*y); }
//	float heading() { return atan2(y, x); }
//	static float dist(Vector2& a, Vector2& b) { return sqrt((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y)); }
//	static float fi(Vector2& a, Vector2& b) { return atan2((b.y - a.y), (b.x - a.x)); }
//};
//Vector2 operator*(Vector2 lhs, float rhs) { return lhs *= rhs; }
//Vector2 operator/(Vector2 lhs, float rhs) { return lhs /= rhs; }
//std::ostream& operator<<(std::ostream& os, const Vector2& obj)
//{
//	os << "(" << obj.x << ", " << obj.y << ")";
//	return os;
//}

float2 randomUnit() { float angle = 2 * M_PI * (rand() % 360) / 360; return{ cosf(angle), sinf(angle) }; }

__host__ __device__ float2 operator+(float2 a, float2 b) { return{ a.x + b.x, a.y + b.y }; }
__host__ __device__ float2& operator+=(float2& lhs, const float2& rhs) { lhs.x += rhs.x; lhs.y += rhs.y; return lhs; }
__host__ __device__ float2& operator-=(float2& lhs, const float2& rhs) { lhs.x -= rhs.x; lhs.y -= rhs.y; return lhs; }
__host__ __device__ float2& operator*=(float2& lhs, float rhs) { lhs.x *= rhs; lhs.y *= rhs; return lhs; }
__host__ __device__ float2& operator/=(float2& lhs, float rhs) { lhs.x /= rhs; lhs.y /= rhs; return lhs; }
__host__ __device__ float2& v_add(float2& lhs, float2& rhs) { lhs += rhs; return lhs; }
__host__ __device__ float2& v_sub(float2& lhs, float2& rhs) { lhs -= rhs; return lhs; }
__host__ __device__ float2& v_mul(float2& lhs, float rhs) { return lhs *= rhs; }
__host__ __device__ float2& v_div(float2& lhs, float rhs) { return lhs /= rhs; }
__host__ __device__ float v_mag(float2& lhs) { return sqrt(lhs.x*lhs.x + lhs.y*lhs.y); }
__host__ __device__ float2& v_norm(float2& lhs) { float m = v_mag(lhs); lhs.x /= m; lhs.y /= m; return lhs; }
__host__ __device__ float2& v_setMag(float2& lhs, float m) { v_norm(lhs); lhs.x *= m; lhs.y *= m; return lhs; }
__host__ __device__ float2& v_limit(float2& lhs, float max) { if (v_mag(lhs) > max) v_setMag(lhs, max); return lhs; }
__host__ __device__ float v_heading(float2& lhs) { return atan2(lhs.y, lhs.x); }
__host__ __device__ float2 v_randomUnit() { float angle = 2 * M_PI * (rand() % 360) / 360; return{ cosf(angle), sinf(angle) }; }
__host__ __device__ float2 v_fromSub(float2& lhs, float2& rhs) { float2 ret(lhs); ret -= rhs; return ret; }
__host__ __device__ float v_dist(float2& a, float2& b) { return sqrt((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y)); }
__host__ __device__ float v_fi(float2& a, float2& b) { return atan2((b.y - a.y), (b.x - a.x)); }
__host__ __device__ float2 operator*(float2 lhs, float rhs) { return lhs *= rhs; }
__host__ __device__ float2 operator/(float2 lhs, float rhs) { return lhs /= rhs; }

struct Shark {
	float2 location;
	Shark() {
		location.x = width / 2.0f;
		location.y = height / 2.0f;
	}
} shark;

struct Fish {
	float2 location;
	float2 velocity;
	float2 acceleration;
	float maxforce;
	float maxspeed;

	Fish() : maxspeed(3.0f), maxforce(0.03f) {
		location.x = width / 2.0f;
		location.y = height / 2.0f;
		float2 random = randomUnit();
		velocity.x = random.x;
		velocity.y = random.y;
		acceleration.x = 0.0f;
		acceleration.y = 0.0f;
	}
	Fish(float x, float y) : maxspeed(3.0f), maxforce(0.03f) {
		location.x = x;
		location.y = y;
		float2 random = randomUnit();
		velocity.x = random.x;
		velocity.y = random.y;
		acceleration.x = 0.0f;
		acceleration.y = 0.0f;
	}

	void run(Fish *fish, Shark *shark, float dt) {
		flock(fish, shark);
		update(dt);
		borders();
	}

	void flock(Fish *fish, Shark *shark) {
		float2 avoidance(avoid(fish));
		float2 alignment(align(fish));
		float2 attraction(attract(fish));
		float2 evade(predator(shark));

		avoidance *= 5.5f;
		alignment *= 1.0f;
		attraction *= 1.0f;
		evade *= 0.05f;

		acceleration += avoidance;
		acceleration += alignment;
		acceleration += attraction;
		acceleration += evade;
	}

	void update(float dt) {
		velocity += acceleration;
		v_limit(velocity, maxspeed);
		location += velocity * dt;
		acceleration *= 0;
	}

	float2 seek(float2 target) {
		// A vector pointing from the location to the target
		float2 desired(v_fromSub(target, location));
		v_setMag(desired, maxspeed);

		float2 steer = v_fromSub(desired, velocity);
		v_limit(steer, maxforce);
		return steer;
	}

	// Wrap-around
	void borders() {
		if (location.x < 0.0f) location.x = width;
		if (location.y < 0.0f) location.y = height;
		if (location.x > width) location.x = 0.0f;
		if (location.y > height) location.y = 0.0f;
	}

	float2 avoid(Fish *fish) {
		float2 steer = { 0.0f, 0.0f };
		int count = 0;
		for (int i = 0; i < FISH_COUNT; ++i) {
			float d = v_dist(location, fish[i].location);
			if ((d > 0) && (d < avoidanceRange)) {
				// Calculate vector pointing away from neighbor
				float2 diff(v_fromSub(location, fish[i].location));
				v_norm(diff);
				v_div(diff, d);		// Weight by distance
				v_add(steer, diff);
				count++;
			}
		}
		// Average
		if (count > 0) {
			v_div(steer, (float)count);
		}

		if (v_mag(steer) > 0) {
			v_setMag(steer, maxspeed);
			v_sub(steer, velocity);
			v_limit(steer, maxforce);
		}
		return steer;
	}

	// Align == avarage velocity
	float2 align(Fish *fish) {
		float2 sum = { 0.0f, 0.0f };
		int count = 0;
		for (int i = 0; i < FISH_COUNT; ++i) {
			float d = v_dist(location, fish[i].location);
			if ((d > 0) && (d < alignmentRange)) {
				v_add(sum, fish[i].velocity);
				count++;
			}
		}
		if (count > 0) {
			v_div(sum, (float)count);
			v_setMag(sum, maxspeed);

			float2 steer = v_fromSub(sum, velocity);
			v_limit(steer, maxforce);
			return steer;
		}
		return { 0.0f, 0.0f };
	}

	// Attraction == avarage location
	float2 attract(Fish *fish) {
		float2 sum = { 0.0f, 0.0f };
		int count = 0;
		for (int i = 0; i < FISH_COUNT; ++i) {
			float d = v_dist(location, fish[i].location);
			if ((d > 0) && (d < attractionRange)) {
				v_add(sum, fish[i].location);
				count++;
			}
		}
		if (count > 0) {
			v_div(sum, count);
			return seek(sum);
		}
		else {
			return { 0.0f, 0.0f };
		}
	}

	float2 predator(Shark *shark) {
		float d = v_dist(location, shark->location);
		if (d > 0 && d < predatorRange) {
			float2 run = v_fromSub(shark->location, location);
			run *= -1;
			v_norm(run);
			v_div(run, d);
			return run;
		}
		return { 0.0f, 0.0f };
	}
} fish[FISH_COUNT];
//std::ostream& operator<<(std::ostream& os, const Fish& obj)
//{
//	os << "Fish: loc " << obj.location << " vel " << obj.velocity;
//	return os;
//}

Fish *d_fish = NULL;
Shark *d_shark = NULL;
float2 *d_acceleration = NULL;

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
int g_Index = 0;
float avgFPS = 0.0f;
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;

int *pArgc = NULL;
char **pArgv = NULL;

#define MAX(a,b) ((a > b) ? a : b)

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
bool run(int argc, char **argv);
void cleanup();
void initState();

// GL functionality
bool initGL(int *argc, char **argv);
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res, unsigned int vbo_res_flags);
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res);

// rendering callbacks
void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void passivemotion(int x, int y);
void timerEvent(int value);

// Cuda functionality
void runCuda(struct cudaGraphicsResource **vbo_resource, struct Fish *d_fish, struct Shark *d_shark);
void runCPU();

void drawCircle(float cx, float cy, float r, int num_segments);
void drawArraw(GLfloat *array, int index, float u, float w, float v, float heading);

const char *sSDKsample = "CUDA fish school";

cudaStream_t stream[4];

///////////////////////////////////////////////////////////////////////////////
//! Simple kernel to modify vertex positions
//! @param data  data in global memory
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

__device__ void draw_arraw(GLfloat *array, int index, float u, float w, float v, float heading)
{
	float peakHeight = 0.02f * d_scale;
	float trunkWidth = 0.005f * d_scale;
	VERT_ARRAY_TRIANGLE(array, index,
		ROTATE_POINT_X(u, w - trunkWidth, u, w, heading),
		ROTATE_POINT_Y(u, w - trunkWidth, u, w, heading),
		v,
		ROTATE_POINT_X(u + peakHeight, w, u, w, heading),
		ROTATE_POINT_Y(u + peakHeight, w, u, w, heading),
		v,
		ROTATE_POINT_X(u, w + trunkWidth, u, w, heading),
		ROTATE_POINT_Y(u, w + trunkWidth, u, w, heading),
		v
	);
}

__device__ float2 seek_func(Fish &subject, float2 target) {
	// A vector pointing from the location to the target
	float2 desired(v_fromSub(target, subject.location));
	v_setMag(desired, subject.maxspeed);

	float2 steer = v_fromSub(desired, subject.velocity);
	v_limit(steer, subject.maxforce);
	return steer;
}

__device__ float2 avoid_func(Fish &subject, Fish *fish) {
	float2 steer = { 0.0f, 0.0f };
	int count = 0;
	for (int i = 0; i < FISH_COUNT; ++i) {
		float d = v_dist(subject.location, fish[i].location);
		if ((d > 0) && (d < d_avoidanceRange)) {
			// Calculate vector pointing away from neighbor
			float2 diff(v_fromSub(subject.location, fish[i].location));
			v_norm(diff);
			v_div(diff, d);		// Weight by distance
			v_add(steer, diff);
			count++;
		}
	}
	// Average
	if (count > 0) {
		v_div(steer, (float)count);
	}

	if (v_mag(steer) > 0) {
		v_setMag(steer, subject.maxspeed);
		v_sub(steer, subject.velocity);
		v_limit(steer, subject.maxforce);
	}
	return steer;
}

// Align == avarage velocity
__device__ float2 align_func(Fish &subject, Fish *fish) {
	float2 sum = { 0.0f, 0.0f };
	int count = 0;
	for (int i = 0; i < FISH_COUNT; ++i) {
		float d = v_dist(subject.location, fish[i].location);
		if ((d > 0) && (d < d_alignmentRange)) {
			v_add(sum, fish[i].velocity);
			count++;
		}
	}
	if (count > 0) {
		v_div(sum, (float)count);
		v_setMag(sum, subject.maxspeed);

		float2 steer = v_fromSub(sum, subject.velocity);
		v_limit(steer, subject.maxforce);
		return steer;
	}
	return{ 0.0f, 0.0f };
}

// Attraction == avarage location
__device__ float2 attract_func(Fish &subject, Fish *fish) {
	float2 sum = { 0.0f, 0.0f };
	int count = 0;
	for (int i = 0; i < FISH_COUNT; ++i) {
		float d = v_dist(subject.location, fish[i].location);
		if ((d > 0) && (d < d_attractionRange)) {
			v_add(sum, fish[i].location);
			count++;
		}
	}
	if (count > 0) {
		v_div(sum, count);
		return seek_func(subject, sum);
	}
	else {
		return{ 0.0f, 0.0f };
	}
}

__device__ float2 predator_func(Fish &subject, Shark *shark) {
	float d = v_dist(subject.location, shark->location);
	if (d > 0 && d < d_predatorRange) {
		float2 run = v_fromSub(shark->location, subject.location);
		run *= -1;
		v_norm(run);
		v_div(run, d);
		return run;
	}
	return { 0.0f, 0.0f };
}

__device__ void flock_func(Fish &subject, Fish *fish, Shark *shark)
{
	subject.acceleration += avoid_func(subject, fish) * 1.5f;
	subject.acceleration += align_func(subject, fish) * 1.0f;
	subject.acceleration += attract_func(subject, fish) * 1.0f;
	subject.acceleration += predator_func(subject, shark) * 0.05f;
}

__device__ void update_func(Fish &subject, float dt, float2 &acceleration)
{
	subject.velocity += acceleration;
	v_limit(subject.velocity, subject.maxspeed);
	subject.location += subject.velocity * dt;
	acceleration *= 0;
}

__device__ void borders_func(Fish &subject)
{
	if (subject.location.x < 0.0f) subject.location.x = d_width;
	if (subject.location.y < 0.0f) subject.location.y = d_height;
	if (subject.location.x > d_width) subject.location.x = 0.0f;
	if (subject.location.y > d_height) subject.location.y = 0.0f;
}

__global__ void run_avoid_kernel(Fish *fish, float2 *acceleration)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i > FISH_COUNT) return;
	acceleration[i] += avoid_func(fish[i], fish) * 1.5f;
}

__global__ void run_align_kernel(Fish *fish, float2 *acceleration)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i > FISH_COUNT) return;
	acceleration[i] += align_func(fish[i], fish) * 1.0f;
}

__global__ void run_attract_kernel(Fish *fish, float2 *acceleration)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i > FISH_COUNT) return;
	acceleration[i] += attract_func(fish[i], fish) * 1.0f;
}

__global__ void run_predator_kernel(Fish *fish, Shark *shark, float2 *acceleration)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i > FISH_COUNT) return;
	acceleration[i] += predator_func(fish[i], shark) * 0.05f;
}

__global__ void run_kernel(GLfloat *pos, Fish *fish, float dt, float2 *acceleration)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i > FISH_COUNT) return;

//	flock_func(fish[i], fish, shark);
	update_func(fish[i], dt, acceleration[i]);
	borders_func(fish[i]);

    // write output vertex
	draw_arraw(pos, i, fish[i].location.x, fish[i].location.y, 1.0f, HEADING(fish[i].velocity));
}


void launch_kernel(GLfloat *pos, Fish *fish, Shark *shark, float dt, float2 *acceleration)
{
    // execute the kernel
	int blockSize = 512;
	dim3 threadsInBlock(blockSize);
	dim3 blocksInGrid(FISH_COUNT / (float)blockSize);
	run_avoid_kernel <<< blocksInGrid, threadsInBlock, 0, stream[0] >>> (fish, acceleration);
	run_align_kernel <<< blocksInGrid, threadsInBlock, 0, stream[1] >>> (fish, acceleration);
	run_attract_kernel <<< blocksInGrid, threadsInBlock, 0, stream[2] >>> (fish, acceleration);
	run_predator_kernel <<< blocksInGrid, threadsInBlock, 0, stream[3] >>> (fish, shark, acceleration);
	checkCudaErrors(cudaDeviceSynchronize());
	run_kernel <<< blocksInGrid, threadsInBlock >>>(pos, fish, dt, acceleration);
	cudaError res = cudaGetLastError();
	if ( cudaSuccess != res )
	    printf( "Error: %s!\n", cudaGetErrorName(res));
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

int findGraphicsGPU(char *name)
{
    int nGraphicsGPU = 0;
    int deviceCount = 0;
    bool bFoundGraphics = false;
    char firstGraphicsName[256], temp[256];

    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess)
    {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
        printf("> FAILED %s sample finished, exiting...\n", sSDKsample);
        exit(EXIT_FAILURE);
    }

    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0)
    {
        printf("> There are no device(s) supporting CUDA\n");
        return false;
    }
    else
    {
        printf("> Found %d CUDA Capable Device(s)\n", deviceCount);
    }

    for (int dev = 0; dev < deviceCount; ++dev)
    {
        bool bGraphics = !checkHW(temp, (const char *)"Tesla", dev);
        printf("> %s\t\tGPU %d: %s\n", (bGraphics ? "Graphics" : "Compute"), dev, temp);

        if (bGraphics)
        {
            if (!bFoundGraphics)
            {
                strcpy(firstGraphicsName, temp);
            }

            nGraphicsGPU++;
        }
    }

    if (nGraphicsGPU)
    {
        strcpy(name, firstGraphicsName);
    }
    else
    {
        strcpy(name, "this hardware");
    }

    return nGraphicsGPU;
}

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

	printf("%s starting...\n", sSDKsample);

	printf("\n");

	run(argc, argv);

	printf("%s completed, returned %s\n", sSDKsample, (g_TotalErrors == 0) ? "OK" : "ERROR!");
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
	glutKeyboardFunc(keyboard);
	glutMotionFunc(motion);
	glutPassiveMotionFunc(passivemotion);
	glutTimerFunc(REFRESH_DELAY, timerEvent, 0);

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
	glOrtho(0.0, width, 0.0, height, 0.1, 10.0);

	SDK_CHECK_ERROR_GL();

	return true;
}


////////////////////////////////////////////////////////////////////////////////
//! Run
////////////////////////////////////////////////////////////////////////////////
bool run(int argc, char **argv/*, char *ref_file*/)
{
	// Create the CUTIL timer
	sdkCreateTimer(&timer);

	// First initialize OpenGL context, so we can properly set the GL for CUDA.
	// This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
	if (false == initGL(&argc, argv))
	{
		return false;
	}

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

	// register callbacks
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutPassiveMotionFunc(passivemotion);
#if defined (__APPLE__) || defined(MACOSX)
	atexit(cleanup);
#else
	glutCloseFunc(cleanup);
#endif

	// create VBO
	createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);

	for (int i = 0; i < 4; ++i)
		checkCudaErrors(cudaStreamCreate(&stream[i]));

	// copy fish & shark to device memory
	int size = FISH_COUNT * sizeof(Fish);
	checkCudaErrors(cudaMalloc((void **) &d_fish, size));
	checkCudaErrors(cudaMemcpy(d_fish, &fish, size, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void **)&d_shark, sizeof(Shark)));
	checkCudaErrors(cudaMemcpy(d_shark, &shark, sizeof(Shark), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void **)&d_acceleration, FISH_COUNT * sizeof(float2)));

	// run the cuda part
	runCuda(&cuda_vbo_resource, d_fish, d_shark);
	//runCPU();

	// start rendering mainloop
	glutMainLoop();

	return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runCuda(struct cudaGraphicsResource **vbo_resource, struct Fish *d_fish, struct Shark *d_shark)
{
    // map OpenGL buffer object for writing from CUDA
    GLfloat *dptr;
    checkCudaErrors(cudaGraphicsMapResources(1, vbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, *vbo_resource));
    //printf("CUDA mapped VBO: May access %ld bytes\n", num_bytes);

	checkCudaErrors(cudaMemcpy(d_shark, &shark, sizeof(Shark), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemset(d_acceleration, 0, FISH_COUNT * sizeof(float2)));

	float dt = 0.013f;

	launch_kernel(dptr, d_fish, d_shark, dt, d_acceleration);

    // unmap buffer object
    checkCudaErrors(cudaGraphicsUnmapResources(1, vbo_resource, 0));
}

void runCPU()
{
	unsigned int size = FISH_COUNT * VERTS_IN_FISH * 4 * sizeof(float);
	GLfloat *pos = (GLfloat*)malloc(size);

	float dt = 0.013f;

	// Compute headings
	for (int i = 0; i < FISH_COUNT; ++i) {
		fish[i].run(fish, &shark, dt);
	}

	// Draw
	for (int i = 0; i < FISH_COUNT; ++i) {
		drawArraw(pos, i, fish[i].location.x, fish[i].location.y, 0.0f, v_heading(fish[i].velocity));
	}

	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, size, pos, GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	free(pos);
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

	// register this buffer object with CUDA
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));

	SDK_CHECK_ERROR_GL();
}

////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res)
{

	// unregister this buffer object with CUDA
	checkCudaErrors(cudaGraphicsUnregisterResource(vbo_res));

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
	runCuda(&cuda_vbo_resource, d_fish, d_shark);
	//runCPU();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// set view matrix
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.0, 0.0, translate_z);
	glRotatef(rotate_x, 1.0, 0.0, 0.0);
	glRotatef(rotate_y, 0.0, 1.0, 0.0);

	// render from the vbo
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glVertexPointer(4, GL_FLOAT, 0, 0);

	glEnableClientState(GL_VERTEX_ARRAY);
	glColor3f(1.0, 0.0, 0.0);
	glDrawArrays(GL_TRIANGLES, 0, FISH_COUNT * VERTS_IN_FISH);
	glDisableClientState(GL_VERTEX_ARRAY);

	glColor3f(0.0f, 1.0f, 0.0f);
	drawCircle(shark.location.x, shark.location.y, predatorRange, 30);

	glutSwapBuffers();

	sdkStopTimer(&timer);
	computeFPS();
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

	for (int i = 0; i < 4; ++i)
		checkCudaErrors(cudaStreamDestroy(stream[i]));

	// cudaDeviceReset causes the driver to clean up all state. While
	// not mandatory in normal operation, it is good practice.  It is also
	// needed to ensure correct operation when the application is being
	// profiled. Calling cudaDeviceReset causes all profile data to be
	// flushed before the application exits
	cudaDeviceReset();
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
	shark.location.x = width * x / (float)window_width;
	shark.location.y = height - height * y / (float)window_height;
}

inline void drawCircle(float cx, float cy, float r, int num_segments)
{
	float theta = 2 * 3.1415926 / float(num_segments);
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

void drawArraw(GLfloat *array, int index, float u, float w, float v, float heading)
{
	//	float peakWidth = 0.02f;
	float peakHeight = 0.02f * scale;
	float trunkWidth = 0.005f * scale;
	//	float trunkHeight = 0.03f;
	VERT_ARRAY_TRIANGLE(array, index,
		ROTATE_POINT_X(u, w - trunkWidth, u, w, heading),
		ROTATE_POINT_Y(u, w - trunkWidth, u, w, heading),
		v,
		ROTATE_POINT_X(u + peakHeight, w, u, w, heading),
		ROTATE_POINT_Y(u + peakHeight, w, u, w, heading),
		v,
		ROTATE_POINT_X(u, w + trunkWidth, u, w, heading),
		ROTATE_POINT_Y(u, w + trunkWidth, u, w, heading),
		v
		);
}
