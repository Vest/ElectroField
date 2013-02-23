#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

/* screen constants */
const int width = 1024;
const int height = 768;

/* charge constants */
const float k = 20.0f;
const float minDistance = 0.1f; // not to divide by zero
const float maxSolidColorLength = 1.0f;

/* charges on the field */
const int maxCharge = 1000;
const int minCharge = -1000;

const char maxChargeCount = 30;
char chargeCount = 0;
__constant__ char dev_chargeCount;

float3 charges[maxChargeCount]; // x, y, z == m
__constant__ float3 dev_charges[maxChargeCount]; // x, y, z == m

/* OpenGL interoperability */
dim3 blocks, threads;

GLuint vbo;
struct cudaGraphicsResource *cuda_vbo_resource;

/* charge selection */
const int detectChargeRange = 20;
int selectedChargeIndex = -1;
bool isDragging = false;

static void HandleError(cudaError_t err, const char *file, int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

void key(unsigned char key, int x, int y) {
	switch (key) {
	case 27:
		printf("Exit application\n");

		glutLeaveMainLoop();
		break;
	}
}

__device__ float length(const float2& q) {
	return sqrtf(q.x * q.x + q.y * q.y);
}

__device__ float length2(const float2& q) {
	return (q.x * q.x + q.y * q.y);
}

__device__ void setColor(const float2& f, uchar4& pixel) {
	pixel.x = pixel.y = pixel.z = pixel.w = 0;

	float l = length(f);
	pixel.x = (l > maxSolidColorLength ? 255 : l * 256 / maxSolidColorLength);
}

__device__ void calculate(const float3& charge, int x, int y, float2& f) {
	f.x = x - charge.x;
	f.y = y - charge.y;

	float l = length2(f) + minDistance;

	float e = charge.z * rsqrt(l * l * l);
	f.x *= e;
	f.y *= e;
}

__global__ void renderFrame(uchar4* screen) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	float2 force, t_force;
	force.x = force.y = 0.0f;

	if (x >= width || y >= height)
		return;

	for (char i = 0; i < dev_chargeCount; i++) {
		calculate(dev_charges[i], x, y, t_force);

		force.x += t_force.x;
		force.y += t_force.y;
	}

	force.x *= k;
	force.y *= k;

	setColor(force, screen[x + y * width]);
}

void idle(void) {
	uchar4* dev_screen;
	size_t size;

	HANDLE_ERROR(cudaGraphicsMapResources(1, &cuda_vbo_resource, 0));
	HANDLE_ERROR(
			cudaGraphicsResourceGetMappedPointer((void**) &dev_screen, &size, cuda_vbo_resource));

	// Kernel Time measure
	cudaEvent_t startEvent, stopEvent;
	float elapsedTime = 0.0f;
	HANDLE_ERROR(cudaEventCreate(&startEvent));
	HANDLE_ERROR(cudaEventCreate(&stopEvent));
	HANDLE_ERROR(cudaEventRecord(startEvent, 0));

	// Render Image
	renderFrame<<<blocks, threads>>>(dev_screen);
	HANDLE_ERROR(cudaDeviceSynchronize());

	HANDLE_ERROR(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));

	// Kernel Time measure
	HANDLE_ERROR(cudaEventRecord(stopEvent, 0));
	HANDLE_ERROR(cudaEventSynchronize(stopEvent));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent));

	char fps[256];
	sprintf(fps, "Electric field: %3.4f ms per frame (FPS: %3.1f)", elapsedTime,
			1000 / elapsedTime);
	glutSetWindowTitle(fps);

	glutPostRedisplay();
}

void draw(void) {
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT);

	glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);

	glPointSize(3.0f);
	glColor3f(0.0f, 1.0f, 1.0f);
	glBegin(GL_POINTS);
	glVertex2i(charges[selectedChargeIndex].x, charges[selectedChargeIndex].y);
	glEnd();

	glutSwapBuffers();
}

void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
		unsigned int vbo_res_flags) {
	unsigned int size = width * height * sizeof(uchar4);

	glGenBuffers(1, vbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, *vbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, size, NULL, GL_DYNAMIC_DRAW);

	HANDLE_ERROR(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));
}

void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res) {
	HANDLE_ERROR(cudaGraphicsUnregisterResource(cuda_vbo_resource));

	glBindBuffer(1, *vbo);
	glDeleteBuffers(1, vbo);

	*vbo = 0;
}

void pushCharge(int x, int y) {
	if (chargeCount < maxChargeCount)
		chargeCount++;
	else {
		for (int i = 0; i < maxChargeCount - 1; ++i) {
			charges[i] = charges[i + 1];
		}
	}

	charges[chargeCount - 1].x = x;
	charges[chargeCount - 1].y = y;
	charges[chargeCount - 1].z = rand() % (maxCharge - minCharge) + minCharge;

	printf("Debug: Charge #%d (%.0f, %.0f, %.0f)\n", chargeCount - 1,
			charges[chargeCount - 1].x, charges[chargeCount - 1].y,
			charges[chargeCount - 1].z);

	HANDLE_ERROR(
			cudaMemcpyToSymbol(dev_charges, charges, chargeCount * sizeof(float3)));
	HANDLE_ERROR(
			cudaMemcpyToSymbol(dev_chargeCount, &chargeCount, sizeof(chargeCount)));
	printf("Charges %d\n", chargeCount);
}

void mouse(int button, int state, int x, int y) {
	if (button != GLUT_LEFT_BUTTON)
		return;

	if (state == GLUT_DOWN) {
		if (selectedChargeIndex != -1) { // Drag
			printf("Drag charge #%d... ", selectedChargeIndex);
			isDragging = true;
		}
	} else {
		if (selectedChargeIndex != -1) { // Drop
			printf("Drop\n");
			isDragging = false;
		} else {
			pushCharge(x, height - y);
		}
	}
}

void mouseDrag(int x, int y) {
	if (isDragging && selectedChargeIndex != -1) {
		printf(" drag... ");
		charges[selectedChargeIndex].x = x;
		charges[selectedChargeIndex].y = height - y;

		HANDLE_ERROR(
				cudaMemcpyToSymbol(dev_charges, charges, chargeCount * sizeof(float3)));
	}
}

void mouseTrack(int x, int y) {
	if (isDragging)
		return;
	// Detect selected charge
	int dx = 0, dy = 0;

	for (int i = 0; i < chargeCount; i++) {
		dx = x - charges[i].x;
		dy = (height - y) - charges[i].y;

		if (dx * dx + dy * dy < detectChargeRange * detectChargeRange) {
			selectedChargeIndex = i;

			return;
		}
	}

	selectedChargeIndex = -1;
}

void initCuda(int deviceId) {
	int deviceCount = 0;
	HANDLE_ERROR(cudaGetDeviceCount(&deviceCount));

	if (deviceCount <= 0) {
		printf("No CUDA devices found\n");
		exit(-1);
	}

	HANDLE_ERROR(cudaGLSetGLDevice(deviceId));

	cudaDeviceProp properties;
	HANDLE_ERROR(cudaGetDeviceProperties(&properties, deviceId));

	threads.x = 32;
	threads.y = properties.maxThreadsPerBlock / threads.x - 2; // to avoid cudaErrorLaunchOutOfResources error

	blocks.x = (width + threads.x - 1) / threads.x;
	blocks.y = (height + threads.y - 1) / threads.y;

	printf(
			"Debug: blocks(%d, %d), threads(%d, %d)\nCalculated Resolution: %d x %d\n",
			blocks.x, blocks.y, threads.x, threads.y, blocks.x * threads.x,
			blocks.y * threads.y);
}

void initGlut(int argc, char** argv) {
	// Initialize freeglut
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowSize(width, height);
	glutCreateWindow("Electric field");
	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);

	glutIdleFunc(idle);
	glutKeyboardFunc(key);
	glutMouseFunc(mouse);
	glutMotionFunc(mouseDrag);
	glutPassiveMotionFunc(mouseTrack);
	glutDisplayFunc(draw);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0.0, (GLdouble) width, 0.0, (GLdouble) height);

	glewInit();
}

int main(int argc, char** argv) {
	setbuf(stdout, NULL);

	initCuda(0);
	initGlut(argc, argv);

	createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);

	glutMainLoop();

	deleteVBO(&vbo, cuda_vbo_resource);

	return 0;
}
