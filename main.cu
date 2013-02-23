#include <stdio.h>
#include <stdlib.h>
#include <GL/freeglut.h>
#include <GL/gl.h>
#include <GL/glext.h>

const int width = 1280;
const int height = 720;

const float k = 50.0f;
const float minDistance = 0.9f; // not to divide by zero
const float maxForce = 1e7f;

uchar4* screen = NULL;

const float maxLengthForColor = 5.0f;

float3 charges[maxChargeCount]; // x, y, z == m
__constant__ float3 dev_charges[maxChargeCount]; // x, y, z == m

struct force {
	float fx, fy;

	__device__ force() :
			fx(0.0f), fy(0.0f) {
	}

	__device__ force(int fx, int fy) :
			fx(fx), fy(fy) {
	}

	__device__ float length2() const {
		return (fx * fx + fy * fy);
	}

	__device__ float length() const {
		return sqrtf(length2());
	}

	__device__ void calculate(const charge& q, int probe_x, int probe_y) {
		// F(1->2) = k * q1 * q2 / r(1->2)^2 * vec_r(1->2) / abs(vec_r(1->2))
		// e = vec_F / q2
		fx = probe_x - q.x;
		fy = probe_y - q.y;

		float l = length();
		if (l <= minDistance) {
			return;
		}

		float e = k * q.q / (l * l * l);
		if (e > maxForce) {
			fx = fy = maxForce;
		} else {
			fx *= e;
			fy *= e;
		}
	}

	__device__ force operator +(const force& f) const {
		return force(fx + f.fx, fy + f.fy);
	}

	__device__ force operator -(const force& f) const {
		return force(fx - f.fx, fy - f.fy);
	}

	__device__ force& operator +=(const force& f) {
		fx += f.fx;
		fy += f.fy;
		return *this;
	}

	__device__ force& operator -=(const force& f) {
		fx -= f.fx;
		fy -= f.fy;
		return *this;
	}
};

const int chargeCount = 10;
__constant__ charge dev_charges[chargeCount];
const int maxCharge = 1000;
const int minCharge = -1000;

int threadsCount(void) {
	int deviceCount = 0;
	cudaGetDeviceCount(&deviceCount);

	if (deviceCount <= 0) {
		printf("No CUDA devices\n");
		exit(-1);
	}

	cudaDeviceProp properties;
	cudaGetDeviceProperties(&properties, 0);

	return properties.maxThreadsPerBlock;
}

void setupGrid(dim3* blocks, dim3* threads, int maxThreads) {
	threads->x = 32;
	threads->y = maxThreads / threads->x - 2; // to avoid cudaErrorLaunchOutOfResources error

	blocks->x = (width + threads->x - 1) / threads->x;
	blocks->y = (height + threads->y - 1) / threads->y;
}

void prepareCharges(void) {
	charge* charges = (charge*) malloc(chargeCount * sizeof(charge));
	for (int i = 0; i < chargeCount; i++) {
		charges[i].x = rand() % width;
		charges[i].y = rand() % height;
		charges[i].q = rand() % (maxCharge - minCharge) + minCharge;

		printf("Debug: Charge #%d (%d, %d, %d)\n", i, charges[i].x,
				charges[i].y, charges[i].q);
	}
	cudaMemcpyToSymbol(dev_charges, charges, chargeCount * sizeof(charge));
}

void key(unsigned char key, int x, int y) {
	switch (key) {
	case 27:
		printf("Exit application\n");
		glutLeaveMainLoop();
		break;
	}
}

void draw(void) {
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT);
	glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, screen);
	glFlush();
}

__device__ uchar4 getColor(const force& f) {
	uchar4 color;
	color.x = color.y = color.z = color.w = 0;

	float l = f.length();
	color.x = (l > maxLengthForColor ? 255 : l * 256 / maxLengthForColor);

	return color;
}

__global__ void renderFrame(uchar4* screen) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	force f, temp_f;
	for (int i = 0; i < chargeCount; i++) {
		temp_f.calculate(dev_charges[i], x, y);
		f += temp_f;
	}

	screen[x + y * width] = getColor(f);
}

int main(int argc, char** argv) {
	// Initialize freeglut
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA);
	glutInitWindowSize(width, height);
	glutCreateWindow("Electric field");
	glutDisplayFunc(draw);
	glutKeyboardFunc(key);
	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);

	// Initialize Host
	dim3 blocks, threads;
	setupGrid(&blocks, &threads, threadsCount());
	printf(
			"Debug: blocks(%d, %d), threads(%d, %d)\nCalculated Resolution: %d x %d\n",
			blocks.x, blocks.y, threads.x, threads.y, blocks.x * threads.x,
			blocks.y * threads.y);

	// Device variables
	screen = (uchar4*) malloc(width * height * sizeof(uchar4));
	memset(screen, 0, width * height * sizeof(uchar4));

	uchar4 *dev_screen = NULL;
	cudaMalloc((void**) &dev_screen, width * height * sizeof(uchar4));
	cudaMemset(dev_screen, 0, width * height * sizeof(uchar4));

	prepareCharges();

	// Launch Kernel to render the image
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);

	renderFrame<<<blocks, threads>>>(dev_screen);

	cudaMemcpy(screen, dev_screen, width * height * sizeof(uchar4),
			cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float elapsedTime = 0.0f;
	cudaEventElapsedTime(&elapsedTime, start, stop);

	printf("Debug: Elapsed time %3.4f ms per frame\n", elapsedTime);

	// Display Image
	glutMainLoop();

	// Free resources
	free(screen);
	screen = NULL;

	cudaFree(dev_screen);
	dev_screen = NULL;

	return 0;
}
