//////////////////////////////////////////////////////////////////////////////////
// This is a front end for a set of viewer clases for the Carnegie Mellon
// Motion Capture Database:
//
//    http://mocap.cs.cmu.edu/
//
// The original viewer code was downloaded from:
//
//   http://graphics.cs.cmu.edu/software/mocapPlayer.zip
//
// where it is credited to James McCann (Adobe), Jernej Barbic (USC),
// and Yili Zhao (USC). There are also comments in it that suggest
// and Alla Safonova (UPenn) and Kiran Bhat (ILM) also had a hand in writing it.
//
//////////////////////////////////////////////////////////////////////////////////

// NOTES:
// -epsilon needs to be pretty large?
//    -strang sphere behavior? (black pixels on boundary)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <float.h>
#include "SETTINGS.h"
#include "skeleton.h"
#include "displaySkeleton.h"
#include "motion.h"
using namespace std;

DisplaySkeleton displayer;
Skeleton *skeleton;
Motion *motion;

int windowWidth = 640;
int windowHeight = 480;
const char *texturePath1 = "woodfloor.ppm";
const char *texturePath2 = "treadplate.ppm";
const char *texturePath3 = "windowpane.ppm";
const int MAX_DEPTH = 2;
const float EPSILON = 1e-3;

int frame = 295;

VEC3 eye(6, 2, -5);
VEC3 eyeCopy = eye;
VEC3 lookingAt(0, 1, 0);
VEC3 up(0, 1, 0);

//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

void readPPM(const string &filename, int &xRes, int &yRes, float *&values)
{
  // try to open the file
  FILE *fp;
  fp = fopen(filename.c_str(), "rb");
  if (fp == NULL)
  {
    cout << " Could not open file \"" << filename.c_str() << "\" for reading." << endl;
    cout << " Make sure you're not trying to read from a weird location or with a " << endl;
    cout << " strange filename. Bailing ... " << endl;
    exit(0);
  }

  // get the dimensions
  unsigned char newline;
  fscanf(fp, "P6\n%d %d\n255%c", &xRes, &yRes, &newline);
  if (newline != '\n')
  {
    cout << " The header of " << filename.c_str() << " may be improperly formatted." << endl;
    cout << " The program will continue, but you may want to check your input. " << endl;
  }
  int totalCells = xRes * yRes;

  // grab the pixel values
  unsigned char *pixels = new unsigned char[3 * totalCells];
  fread(pixels, 1, totalCells * 3, fp);

  // copy to a nicer data type
  values = new float[3 * totalCells];
  for (int i = 0; i < 3 * totalCells; i++)
    values[i] = pixels[i];

  // clean up
  delete[] pixels;
  fclose(fp);
  cout << " Read in file " << filename.c_str() << endl;
}

void writePPM(const string &filename, int &xRes, int &yRes, const float *values)
{
  int totalCells = xRes * yRes;
  unsigned char *pixels = new unsigned char[3 * totalCells];
  for (int i = 0; i < 3 * totalCells; i++)
    pixels[i] = values[i];

  FILE *fp;
  fp = fopen(filename.c_str(), "wb");
  if (fp == NULL)
  {
    std::cout << " Could not open file \"" << filename.c_str() << "\" for writing." << endl;
    std::cout << " Make sure you're not trying to write from a weird location or with a " << endl;
    std::cout << " strange filename. Bailing ... " << endl;
    exit(0);
  }

  fprintf(fp, "P6\n%d %d\n255\n", xRes, yRes);
  fwrite(pixels, 1, totalCells * 3, fp);
  fclose(fp);
  delete[] pixels;
}

struct Ray
{
  VEC3 origin;
  VEC3 direction;

  Ray() : origin(VEC3(0.0, 0.0, 0.0)), direction(VEC3(0.0, 0.0, 0.0)) {}
  Ray(const VEC3 &o, const VEC3 &d) : origin(o), direction(d) {}
};

struct Sphere
{
  VEC3 center;
  double radius;
  VEC3 color;

  Sphere() : center(VEC3(0.0, 0.0, 0.0)), radius(0.0), color(VEC3(0.0, 0.0, 0.0)) {}
  Sphere(const VEC3 &cen, double r, const VEC3 &col) : center(cen), radius(r), color(col) {}
};

struct Triangle
{
  VEC3 v0;
  VEC3 v1;
  VEC3 v2;
  VEC3 color;
  int textureID;

  Triangle(const VEC3 &v0, const VEC3 &v1, const VEC3 &v2, const VEC3 &color, int textureID) : v0(v0), v1(v1), v2(v2), color(color), textureID(textureID) {}
};

struct Cylinder
{
  VEC3 leftVertex;
  VEC3 rightVertex;
  VEC3 direction;
  double radius;
  VEC3 color;

  Cylinder(const VEC3 &leftVertex, const VEC3 &rightVertex, const VEC3 &direction, double radius, const VEC3 &color)
      : leftVertex(leftVertex), rightVertex(rightVertex), direction(direction.normalized()), radius(radius), color(color) {}
};

struct PointLight
{
  VEC3 position;
  VEC3 color;

  PointLight(const VEC3 &position, const VEC3 &color) : position(position), color(color) {}
};

struct Texture
{
  int width;
  int height;
  vector<VEC3> data; // Assuming VEC3 is a vector class or struct
};

struct HitRecord
{
  float t; // Parameter along the ray
  float t2;
  VEC3 hitPoint; // Point of intersection
  VEC3 normal;   // Surface normal at the hit point
  float u;       // Barycentric coordinate u
  float v;       // Barycentric coordinate v
  float w;       // Barycentric coordinate w
  VEC3 color;

  HitRecord() : t(0.0), t2(0.0), hitPoint(VEC3(0.0, 0.0, 0.0)), normal(VEC3(0.0, 0.0, 0.0)), u(0.0f), v(0.0f), w(0.0f), color(VEC3(0.0, 0.0, 0.0)) {}

  HitRecord(float _t, float _t2, const VEC3 &_hitPoint, const VEC3 &_normal, float _u, float _v, float _w, const VEC3 &_color)
      : t(_t), t2(_t2), hitPoint(_hitPoint), normal(_normal), u(_u), v(_v), w(_w), color(_color) {}
};

// more scene geometry
vector<Texture> textures;
vector<Ray> rays;
vector<Sphere> spheres;
vector<Triangle> triangles;
vector<Cylinder> cylinders;
vector<PointLight> lights;

//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

VEC3 componentWiseMultiply(const VEC3 &vec1, const VEC3 &vec2)
{
  double x = vec1.x() * vec2.x();
  double y = vec1.y() * vec2.y();
  double z = vec1.z() * vec2.z();
  return VEC3(x, y, z);
}

void setUpTexture(const char *texturePath)
{
  Texture texture;

  int textureWidth, textureHeight;
  float *textureData;

  // Assuming texturePath contains the path to the PPM file
  readPPM(texturePath, textureWidth, textureHeight, textureData);

  texture.width = textureWidth;
  texture.height = textureHeight;
  texture.data.resize(textureWidth * textureHeight, VEC3(0, 0, 0));

  for (int i = 0; i < textureWidth * textureHeight * 3; i += 3)
  {
    // Assuming each channel is a float (0.0-255.0)
    float r = textureData[i] / 255.0f;
    float g = textureData[i + 1] / 255.0f;
    float b = textureData[i + 2] / 255.0f;

    if (strcmp(texturePath, "windowpane.ppm") == 0)
    {
      r /= 1.60;
      g /= 1.60;
      b /= 1.60;
    }

    texture.data[i / 3] = VEC3(r, g, b);
  }

  textures.push_back(texture);
  // Release textureData resources
  delete[] textureData;
}

void createCoordinateSystem(const VEC3 &up, VEC3 &tangent, VEC3 &bitangent)
{
  if (fabs(up.x()) > fabs(up.y()))
  {
    // Normalize the tangent vector
    float invLength = 1.0f / sqrt(up.x() * up.x() + up.z() * up.z());
    tangent = VEC3(-up.z() * invLength, 0.0f, up.x() * invLength);
  }
  else
  {
    // Normalize the tangent vector
    float invLength = 1.0f / sqrt(up.y() * up.y() + up.z() * up.z());
    tangent = VEC3(0.0f, up.z() * invLength, -up.y() * invLength);
  }

  // Calculate the bitangent vector
  bitangent = up.cross(tangent);
}

VEC3 generateRandomDirection(const VEC3 &normal)
{
  double theta = 2.0 * M_PI * (rand() / (double)RAND_MAX);
  double phi = acos(1.0 - 2.0 * (rand() / (double)RAND_MAX));

  double x = sin(phi) * cos(theta);
  double y = sin(phi) * sin(theta);
  double z = cos(phi);

  // Transform the random direction to the local coordinate system (normal as the up direction)
  VEC3 tangent, bitangent;
  createCoordinateSystem(normal, tangent, bitangent);

  return x * tangent + y * bitangent + z * normal;
}

VEC3 calculateNormal(const VEC3 &hitPoint, int hitID)
{
  if (hitID < spheres.size())
  {
    // Sphere
    const Sphere &sphere = spheres[hitID];
    return (hitPoint - sphere.center).normalized();
  }
  else if (hitID < spheres.size() + triangles.size())
  {
    // Triangle
    const Triangle &triangle = triangles[hitID - spheres.size()];
    VEC3 e1 = triangle.v1 - triangle.v0;
    VEC3 e2 = triangle.v2 - triangle.v0;
    return (e1.cross(e2)).normalized();
  }
  else
  {
    // Cylinder
    const Cylinder &cylinder = cylinders[hitID - spheres.size() - triangles.size()];
    VEC3 axis = (cylinder.rightVertex - cylinder.leftVertex).normalized();
    VEC3 hitPointOnCylinder = cylinder.leftVertex + axis * (axis.dot(hitPoint - cylinder.leftVertex));
    return (hitPoint - hitPointOnCylinder).normalized();
  }
}

VEC3 computeFacingForwardNormal(const VEC3 &normal, const VEC3 &rayDirection)
{
  return (normal.dot(rayDirection) > 0) ? -1 * normal : 1 * normal;
}

// Function to compute the reflection direction for an incident ray hitting a surface with a given normal
VEC3 reflect(const VEC3 &incident, const VEC3 &normal)
{
  return incident - 2.0 * incident.dot(normal) * normal;
}

// Function to compute the refraction direction for an incident ray hitting a surface with a given normal
VEC3 refract(const VEC3 &incident, const VEC3 &normal, float refractiveIndex)
{
  float cosTheta = -incident.dot(normal);
  float k = 1.0 - refractiveIndex * refractiveIndex * (1.0 - cosTheta * cosTheta);

  // Check for total internal reflection
  if (k < 0.0)
  {
    return VEC3(0.0, 0.0, 0.0); // Total internal reflection
  }
  else
  {
    return refractiveIndex * incident + (refractiveIndex * cosTheta - sqrt(k)) * normal;
  }
}

VEC3 computeDiffuse(const VEC3 &normal, const VEC3 &lightDir, const VEC3 &lightColor)
{
  float cosTheta = std::max(0.0, normal.dot(lightDir));
  return lightColor * cosTheta;
}

VEC3 sampleTexture(Texture texture, float u, float v)
{
  // Assuming texture is a 2D array of VEC3 representing RGB values
  int x = static_cast<int>((u * texture.width)) % texture.width;
  int y = static_cast<int>((v * texture.height)) % texture.height;

  // Sample the texture color at the specified coordinates
  return texture.data[x + y * texture.width];
}

bool raySphereIntersect(const Ray &ray, const Sphere &sphere, HitRecord &hitRecord)
{
  const VEC3 op = sphere.center - ray.origin;
  const float b = op.dot(ray.direction);
  float discriminant = b * b - op.dot(op) + sphere.radius * sphere.radius;

  // Determinant check
  if (discriminant < 0)
    return false;

  discriminant = sqrt(discriminant);
  hitRecord.t = b - discriminant;
  if (hitRecord.t <= EPSILON)
  {
    hitRecord.t = b + discriminant;
    if (hitRecord.t <= EPSILON)
      hitRecord.t = -1;
  }

  if (hitRecord.t < 0)
    return false;
  return true;
}

bool rayTriangleIntersect(const Ray &ray, const Triangle &triangle, HitRecord &hitRecord)
{
  double denominator = ray.direction.dot(hitRecord.normal);
  double epsilon = 1e-8;

  if (fabs(denominator) <= epsilon)
  {
    return false;
  }

  hitRecord.t = (triangle.v0 - ray.origin).dot(hitRecord.normal) / denominator;

  if (hitRecord.t <= epsilon)
  {
    return false;
  }

  hitRecord.hitPoint = ray.origin + hitRecord.t * ray.direction;

  VEC3 edge0 = triangle.v0 - hitRecord.hitPoint;
  VEC3 edge1 = triangle.v1 - hitRecord.hitPoint;
  VEC3 edge2 = triangle.v2 - hitRecord.hitPoint;

  if (hitRecord.normal.dot(edge0.cross(edge1)) >= 0 &&
      hitRecord.normal.dot(edge1.cross(edge2)) >= 0 &&
      hitRecord.normal.dot(edge2.cross(edge0)) >= 0)
  {

    return true;
  }

  return false;
}

bool rayCylinderIntersect(const Ray &ray, const Cylinder &cylinder, HitRecord &hitRecord)
{
  double epsilon = 1e-8;
  // Translate and rotate the ray and cylinder to local coordinates
  VEC3 o = ray.origin;
  VEC3 d = ray.direction;

  // Cylinder parameters in local coordinates
  VEC3 p0 = cylinder.leftVertex;
  VEC3 axis = cylinder.rightVertex - cylinder.leftVertex;
  VEC3 axisDirection = axis.normalized();
  double radius = cylinder.radius;

  // Helper vector
  VEC3 oc = o - p0;
  VEC3 crossDir = d.cross(axisDirection);
  VEC3 crossDir2 = oc.cross(axisDirection);

  // Coefficients of the quadratic equation
  float a = crossDir.dot(crossDir);
  float b = 2.0f * crossDir.dot(crossDir2);
  float c = crossDir2.dot(crossDir2) - pow(radius, 2);

  // Compute the discriminant
  float discriminant = b * b - 4 * a * c;

  if (discriminant <= epsilon)
  {
    return false;
  }

  // Compute the solutions to the quadratic equation
  hitRecord.t = (-b - sqrt(discriminant)) / (2.0f * a);
  hitRecord.t2 = (-b + sqrt(discriminant)) / (2.0f * a);

  // Check if the intersections are within the bounds of the cylinder
  VEC3 intersection1 = o + hitRecord.t * d;
  VEC3 intersection2 = o + hitRecord.t2 * d;

  float dot1 = (intersection1 - p0).dot(axisDirection);
  float dot2 = (intersection2 - p0).dot(axisDirection);

  // Check if the intersection points are within the bounds of the cylinder
  if (dot1 >= 0 && dot1 <= axis.norm() && hitRecord.t >= 0)
  {
    return true;
  }

  if (dot2 >= 0 && dot2 <= axis.norm() && hitRecord.t2 >= 0)
  {
    return true;
  }

  return false;
}

bool isInShadowSphere(const VEC3 &point, const PointLight &light, const vector<Sphere> &spheres)
{
  VEC3 lightDir = (light.position - point).normalized();
  Ray shadowRay(point, lightDir);

  for (size_t i = 0; i < spheres.size(); ++i)
  {
    HitRecord hitRecord;
    if (raySphereIntersect(shadowRay, spheres[i], hitRecord))
    {
      return true;
    }
  }

  return false;
}

bool isInShadowTriangle(const VEC3 &point, const PointLight &light, const vector<Triangle> &triangles)
{
  VEC3 lightDir = (light.position - point).normalized();
  Ray shadowRay(point, lightDir);

  for (size_t i = 0; i < triangles.size(); ++i)
  {
    VEC3 e1 = triangles[i].v1 - triangles[i].v0;
    VEC3 e2 = triangles[i].v2 - triangles[i].v0;
    VEC3 normal = e1.cross(e2).normalized();

    HitRecord hitRecord;
    hitRecord.normal = normal;
    if (rayTriangleIntersect(shadowRay, triangles[i], hitRecord))
    {
      return true;
    }
  }

  return false;
}

bool isInShadowCylinder(const VEC3 &point, const PointLight &light, const vector<Cylinder> &cylinders)
{
  VEC3 lightDir = (light.position - point).normalized();
  Ray shadowRay(point, lightDir);

  for (size_t i = 0; i < cylinders.size(); ++i)
  {
    HitRecord hitRecord;
    if (rayCylinderIntersect(shadowRay, cylinders[i], hitRecord))
    {
      return true;
    }
  }

  return false;
}

VEC3 diffuseShading(const VEC3 &lightDir, const VEC3 &normal, const VEC3 &sphereColor, const VEC3 &lightColor)
{
  VEC3 normalizedLightDir = lightDir.normalized();
  VEC3 normalizedNormal = normal.normalized();
  double lambertian = normalizedLightDir.dot(normalizedNormal);
  lambertian = max(lambertian, 0.0);
  VEC3 diffuse = componentWiseMultiply(sphereColor, lightColor) * lambertian;
  return diffuse;
}

VEC3 specularShading(const VEC3 &lightDir, const VEC3 &normal, const VEC3 &viewDir, const VEC3 &sphereColor, const VEC3 &lightColor, double shininess)
{
  VEC3 normalizedLightDir = lightDir.normalized();
  VEC3 normalizedNormal = normal.normalized();
  VEC3 normalizedViewDir = viewDir.normalized();
  VEC3 reflectionDir = (2.0 * normalizedLightDir.dot(normalizedNormal) * normalizedNormal - normalizedLightDir).normalized();
  double specular = pow(max(reflectionDir.dot(normalizedViewDir), 0.0), shininess);
  specular = max(specular, 0.0);
  VEC3 specularColor = componentWiseMultiply(sphereColor, lightColor) * specular;
  return specularColor;
}

VEC3 traceRay(const Ray &ray, int depth)
{

  if (depth <= 0)
  {
    return VEC3(0, 0, 0); // Reached recursion limit, return black
  }

  VEC3 pixelColor(0, 0, 0);
  HitRecord hitRecord;
  HitRecord hitRecordFinal;
  int hitID = -1;
  float tMinFound = FLT_MAX;

  // Check for sphere intersections
  for (int i = 0; i < spheres.size(); i++)
  {
    hitRecord.t = FLT_MAX;
    if (raySphereIntersect(ray, spheres[i], hitRecord) && hitRecord.t < tMinFound)
    {
      tMinFound = hitRecord.t;
      hitID = i;
    }
  }

  // Check for triangle intersections
  for (int i = 0; i < triangles.size(); i++)
  {
    VEC3 e1 = triangles[i].v1 - triangles[i].v0;
    VEC3 e2 = triangles[i].v2 - triangles[i].v0;
    VEC3 normal = e1.cross(e2).normalized();
    hitRecord.normal = normal;
    hitRecord.t = FLT_MAX;
    if (rayTriangleIntersect(ray, triangles[i], hitRecord))
    {
      if (hitRecord.t < tMinFound)
      {
        tMinFound = hitRecord.t;
        hitID = i + spheres.size();
        if (triangles[i].textureID != -1)
        {
          // Perform texture sampling directly within rayTriangleIntersect
          float areaABC = hitRecord.normal.dot((triangles[i].v1 - triangles[i].v0).cross(triangles[i].v2 - triangles[i].v0));
          float areaPBC = hitRecord.normal.dot((triangles[i].v1 - hitRecord.hitPoint).cross(triangles[i].v2 - hitRecord.hitPoint));
          float areaPCA = hitRecord.normal.dot((triangles[i].v2 - hitRecord.hitPoint).cross(triangles[i].v0 - hitRecord.hitPoint));

          hitRecord.u = areaPBC / areaABC;
          hitRecord.v = areaPCA / areaABC;
          hitRecord.w = 1.0f - hitRecord.u - hitRecord.v;

          // Sample texture and assign to hitRecord.color
          hitRecord.color = sampleTexture(textures[triangles[i].textureID], hitRecord.u, hitRecord.v);
        }
        else
        {
          hitRecord.color = triangles[i].color;
        }
      }
    }
  }

  // Check for cylinder intersections
  for (int i = 0; i < cylinders.size(); i++)
  {
    hitRecord.t = FLT_MAX;
    hitRecord.t2 = FLT_MAX;
    if (rayCylinderIntersect(ray, cylinders[i], hitRecord))
    {
      // Choose the closest valid intersection point
      if (hitRecord.t > 0 && hitRecord.t < tMinFound)
      {
        tMinFound = hitRecord.t;
        hitID = i + spheres.size() + triangles.size();
      }
      if (hitRecord.t2 > 0 && hitRecord.t2 < tMinFound)
      {
        tMinFound = hitRecord.t2;
        hitID = i + spheres.size() + triangles.size();
        ;
      }
    }
  }

  if (hitID != -1)
  {
    VEC3 hitPoint = ray.origin + tMinFound * ray.direction;
    VEC3 modifiedHitPoint = ray.origin + tMinFound * ray.direction - EPSILON * ray.direction;
    for (int i = 0; i < lights.size(); i++)
    {
      VEC3 lightDir = (lights[i].position - hitPoint).normalized();
      VEC3 viewDir = -ray.direction.normalized();
      if (hitID < spheres.size())
      {
        hitRecord.normal = calculateNormal(hitPoint, hitID);
        VEC3 diffuse = diffuseShading(lightDir, hitRecord.normal, spheres[hitID].color, lights[i].color);
        VEC3 specular = specularShading(lightDir, hitRecord.normal, viewDir, spheres[hitID].color, lights[i].color, 32.0);
        pixelColor += diffuse + specular;

        if (hitID == 0)
        {
          // Calculate glossy reflection
          const int numSamples = 100;
          VEC3 accumulatedColor = spheres[hitID].color;

          for (int i = 0; i < numSamples; ++i)
          {
            VEC3 reflectionDir = reflect(ray.direction, hitRecord.normal);
            VEC3 modifiedRelflectionDir = (reflectionDir + 0.09 * generateRandomDirection(hitRecord.normal)).normalized();

            // Recursive call for the reflected ray
            Ray reflectedRay(hitPoint + reflectionDir * EPSILON, modifiedRelflectionDir);
            accumulatedColor += traceRay(reflectedRay, depth - 1);
          }
          // Average the accumulated colors
          pixelColor += accumulatedColor / numSamples;
        }
        else
        {
          pixelColor += spheres[hitID].color;
        }
      }
      else if (hitID < spheres.size() + triangles.size())
      {
        hitRecord.normal = calculateNormal(hitPoint, hitID);
        hitRecord.normal = computeFacingForwardNormal(hitRecord.normal, ray.direction);
        VEC3 diffuse = diffuseShading(lightDir, hitRecord.normal, VEC3(0.5, 0.5, 0.5), lights[i].color);
        pixelColor += diffuse;
        pixelColor += hitRecord.color;
      }
      else
      {
        hitRecord.normal = calculateNormal(hitPoint, hitID);
        hitRecord.normal = computeFacingForwardNormal(hitRecord.normal, ray.direction);
        VEC3 diffuse = diffuseShading(lightDir, hitRecord.normal, cylinders[hitID - spheres.size() - triangles.size()].color, lights[i].color);
        VEC3 specular = specularShading(lightDir, hitRecord.normal, viewDir, cylinders[hitID - spheres.size() - triangles.size()].color, lights[i].color, 32.0);
        pixelColor += diffuse + specular;
      }

      if (isInShadowSphere(modifiedHitPoint, lights[i], spheres) || isInShadowTriangle(modifiedHitPoint, lights[i], triangles) || isInShadowCylinder(modifiedHitPoint, lights[i], cylinders))
      {
        pixelColor *= 0.3;
      }
    }
  }

  return pixelColor;
}

//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

void rayColor(const Ray &ray, VEC3 &pixelColor)
{
  pixelColor += traceRay(ray, MAX_DEPTH);
}

//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
float clamp(float value)
{
  if (value < 0.0)
    return 0.0;
  else if (value > 1.0)
    return 1.0;
  return value;
}

//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
void renderImage(int &xRes, int &yRes, const string &filename)
{
  // Allocate the final image
  const int totalCells = xRes * yRes;
  float *ppmOut = new float[3 * totalCells];

  if (frame < 295)
  {
    eye.z() = eyeCopy.z() + 0.034 * frame;
    eye.x() = eyeCopy.x() - 0.02 * frame;
    eye.y() = eyeCopy.y() + 0.001 * frame;
  }
  else
  {
    eye.z() = eyeCopy.z() + 0.034 * 294;
    eye.x() = eyeCopy.x() - 0.02 * 294;
    eye.y() = eyeCopy.y() + 0.001 * frame;
  }

  // Compute image plane
  const float halfY = (lookingAt - eye).norm() * tan(45.0f / 360.0f * M_PI);
  const float halfX = halfY * 4.0f / 3.0f;

  // Get rhand translation
  // vector<VEC4> &translations = displayer.translations();
  // VEC4 boneTranslation = translations[17];
  // VEC3 bonePosition = boneTranslation.head<3>() * 2;

  // Update camera positions based on pelvis position
  // VEC3 translatedEye = (bonePosition.x() + eye.x(), bonePosition.y() + eye.y(), bonePosition.z() + eye.z());
  // VEC3 translatedLookingAt = VEC3(bonePosition.x() + lookingAt.x(), bonePosition.y() + lookingAt.y(), bonePosition.z() + lookingAt.z());

  VEC3 translatedEye = eye;
  VEC3 translatedLookingAt = lookingAt;

  const VEC3 cameraZ = (lookingAt - translatedEye).normalized();
  const VEC3 cameraX = up.cross(cameraZ).normalized();
  const VEC3 cameraY = cameraZ.cross(cameraX).normalized();

  for (int y = 0; y < yRes; y++)
    for (int x = 0; x < xRes; x++)
    {
      const float ratioX = 1.0f - x / float(xRes) * 2.0f;
      const float ratioY = 1.0f - y / float(yRes) * 2.0f;
      const VEC3 rayHitImage = lookingAt +
                               ratioX * halfX * cameraX +
                               ratioY * halfY * cameraY;
      const VEC3 rayDir = (rayHitImage - eye).normalized();
      const Ray ray(eye, rayDir);

      VEC3 color(0, 0, 0);
      rayColor(ray, color);

      ppmOut[3 * (y * xRes + x)] = clamp(color[0]) * 255.0f;
      ppmOut[3 * (y * xRes + x) + 1] = clamp(color[1]) * 255.0f;
      ppmOut[3 * (y * xRes + x) + 2] = clamp(color[2]) * 255.0f;
    }

  frame++;

  writePPM(filename, xRes, yRes, ppmOut);

  delete[] ppmOut;
}

//////////////////////////////////////////////////////////////////////////////////
// Load up a new motion captured frame
//////////////////////////////////////////////////////////////////////////////////
void setSkeletonsToSpecifiedFrame(int frameIndex)
{
  if (frameIndex < 0)
  {
    printf("Error in SetSkeletonsToSpecifiedFrame: frameIndex %d is illegal.\n", frameIndex);
    exit(0);
  }
  if (displayer.GetSkeletonMotion(0) != NULL)
  {
    int postureID;
    if (frameIndex >= displayer.GetSkeletonMotion(0)->GetNumFrames())
    {
      std::cout << " We hit the last frame! You might want to pick a different sequence. " << endl;
      postureID = displayer.GetSkeletonMotion(0)->GetNumFrames() - 1;
    }
    else
      postureID = frameIndex;
    displayer.GetSkeleton(0)->setPosture(*(displayer.GetSkeletonMotion(0)->GetPosture(postureID)));
  }
}

//////////////////////////////////////////////////////////////////////////////////
// Build a list of spheres in the scene
//////////////////////////////////////////////////////////////////////////////////
void buildScene()
{
  spheres.clear();
  triangles.clear();
  cylinders.clear();
  lights.clear();
  displayer.ComputeBonePositions(DisplaySkeleton::BONES_AND_LOCAL_FRAMES);

  // Retrieve all the bones of the skeleton
  vector<MATRIX4> &rotations = displayer.rotations();
  vector<MATRIX4> &scalings = displayer.scalings();
  vector<VEC4> &translations = displayer.translations();
  vector<float> &lengths = displayer.lengths();

  // Add lights
  PointLight light1(VEC3(8, 5, -5), VEC3(1, 1, 1));
  PointLight light2(VEC3(-8, 5, -5), VEC3(1, 1, 1));
  lights.push_back(light1);
  // lights.push_back(light2);

  // Add a checkerboard floor
  VEC3 floorV0(-5, 0, 5);
  VEC3 floorColor1(0, 0, 1);
  VEC3 floorColor2(0, 1, 1);
  int floorLength = 10;
  int divisions = 5;
  int sideLength = floorLength / divisions;
  // Loop to create the checkerboard pattern
  for (int i = 0; i < floorLength; i += sideLength)
  {
    for (int j = 0; j < floorLength; j += sideLength)
    {
      // Calculate the vertices for each square
      VEC3 v0 = floorV0 + VEC3(i, 0, -j);
      VEC3 v1 = floorV0 + VEC3(i + sideLength, 0, -j);
      VEC3 v2 = floorV0 + VEC3(i, 0, -(j + sideLength));
      VEC3 v3 = floorV0 + VEC3(i + sideLength, 0, -(j + sideLength));

      // Alternate between floorColor1 and floorColor2 for checkerboard pattern
      VEC3 color = ((i / sideLength + j / sideLength) % 2 == 0) ? floorColor1 : floorColor2;

      // Create and push back two triangles for each square
      triangles.push_back(Triangle(v1, v2, v0, color, 0));
      triangles.push_back(Triangle(v2, v1, v3, color, 0));
    }
  }

  VEC3 v1(-5, 0, 5);
  VEC3 v2(5, 0, 5);
  VEC3 v3(5, 5, 5);
  VEC3 color(101 / 255.0, 67 / 255.0, 33 / 255.0);
  triangles.push_back(Triangle(v1, v2, v3, color, -1));

  v1 = VEC3(-5, 5, 5);
  v2 = VEC3(-5, 0, 5);
  v3 = VEC3(5, 5, 5);
  triangles.push_back(Triangle(v1, v2, v3, color, -1));

  v1 = VEC3(-5, 5, 5);
  v2 = VEC3(-5, 0, 5);
  v3 = VEC3(5, 5, 5);
  triangles.push_back(Triangle(v1, v2, v3, color, -1));

  v1 = VEC3(-5, 5, 5);
  v2 = VEC3(-5, 0, -5);
  v3 = VEC3(-5, 0, 5);
  triangles.push_back(Triangle(v1, v2, v3, color, -1));

  v1 = VEC3(-5, 5, 5);
  v2 = VEC3(-5, 0, -5);
  v3 = VEC3(-5, 5, -5);
  triangles.push_back(Triangle(v1, v2, v3, color, -1));

  // Define the color for the squares
  VEC3 squareColor = VEC3(1, 0, 0); // Red color, change as needed

  // Offset value
  float offset = 0.0009;
  float squareSize = 1.5; // Adjust the size of the squares as needed

  // Create the first square
  v1 = VEC3(-5 + offset, 2.5 + squareSize / 2, 3);
  v2 = VEC3(-5 + offset, 2.5 - squareSize / 2, 3);
  v3 = VEC3(-5 + offset, 2.5 - squareSize / 2, 3 - squareSize);
  triangles.push_back(Triangle(v1, v3, v2, squareColor, 2));

  v1 = VEC3(-5 + offset, 2.5 + squareSize / 2, 3);
  v2 = VEC3(-5 + offset, 2.5 - squareSize / 2, 3 - squareSize);
  v3 = VEC3(-5 + offset, 2.5 + squareSize / 2, 3 - squareSize);
  triangles.push_back(Triangle(v1, v2, v3, squareColor, 2));

  // Create the second square
  v1 = VEC3(-5 + offset, 2.5 + squareSize / 2, 0);
  v2 = VEC3(-5 + offset, 2.5 - squareSize / 2, 0);
  v3 = VEC3(-5 + offset, 2.5 - squareSize / 2, 0 - squareSize);
  triangles.push_back(Triangle(v1, v3, v2, squareColor, 2));

  v1 = VEC3(-5 + offset, 2.5 + squareSize / 2, 0);
  v2 = VEC3(-5 + offset, 2.5 - squareSize / 2, 0 - squareSize);
  v3 = VEC3(-5 + offset, 2.5 + squareSize / 2, 0 - squareSize);
  triangles.push_back(Triangle(v1, v2, v3, squareColor, 2));

  // Add a sphere
  VEC3 orbCenter(0, 2, 1.9);
  double orbRadius = 1.0;
  VEC3 orbColor(1, 215 / 255.0, 0.1);
  spheres.push_back(Sphere(orbCenter, orbRadius, orbColor));

  // Define the vertices of the rectangular prism
  double prismHeight = 1.2;
  double prismHalfWidth = 0.8;
  VEC3 boxV0(-prismHalfWidth, 0, orbCenter.z() - sideLength / 2.0);           // Bottom-left back corner
  VEC3 boxV1(-prismHalfWidth, prismHeight, orbCenter.z() - sideLength / 2.0); // Top-left back corner
  VEC3 boxV2(prismHalfWidth, 0, orbCenter.z() - sideLength / 2.0);            // Bottom-right back corner
  VEC3 boxV3(prismHalfWidth, prismHeight, orbCenter.z() - sideLength / 2.0);  // Top-right back corner
  VEC3 boxV4(-prismHalfWidth, prismHeight, orbCenter.z() + sideLength / 2.0); // Top-left front corner
  VEC3 boxV5(prismHalfWidth, prismHeight, orbCenter.z() + sideLength / 2.0);  // Top-right front corner
  VEC3 boxV6(-prismHalfWidth, 0, orbCenter.z() + sideLength / 2.0);           // Bottom-left front corner
  VEC3 boxV7(prismHalfWidth, 0, orbCenter.z() + sideLength / 2.0);            // Bottom-right front corner

  // Define the color for the rectangular prism
  VEC3 boxColor(192 / 255.0, 192 / 255.0, 192 / 255.0); // Light grey

  // Create and push back the triangles for the rectangular prism
  triangles.push_back(Triangle(boxV4, boxV3, boxV1, boxColor, 1)); // Top face
  triangles.push_back(Triangle(boxV3, boxV4, boxV5, boxColor, 1));
  triangles.push_back(Triangle(boxV5, boxV6, boxV4, boxColor, 1)); // Front face
  triangles.push_back(Triangle(boxV6, boxV5, boxV7, boxColor, 1));
  triangles.push_back(Triangle(boxV2, boxV1, boxV0, boxColor, 1)); // Back face
  triangles.push_back(Triangle(boxV1, boxV2, boxV3, boxColor, 1));
  triangles.push_back(Triangle(boxV6, boxV1, boxV0, boxColor, 1)); // Left face
  triangles.push_back(Triangle(boxV1, boxV6, boxV4, boxColor, 1));
  triangles.push_back(Triangle(boxV7, boxV3, boxV2, boxColor, 1)); // Right face
  triangles.push_back(Triangle(boxV3, boxV7, boxV5, boxColor, 1));

  // build a sphere list, but skip the first bone,
  // it's just the origin
  int totalBones = rotations.size();
  for (int x = 1; x < totalBones; x++)
  {
    MATRIX4 &rotation = rotations[x];
    MATRIX4 &scaling = scalings[x];
    VEC4 &translation = translations[x];

    // get the endpoints of the cylinder
    VEC4 leftVertex(0, 0, 0, 1);
    VEC4 rightVertex(0, 0, lengths[x], 1);

    leftVertex = rotation * scaling * leftVertex + translation;
    rightVertex = rotation * scaling * rightVertex + translation;

    // get the direction vector
    VEC3 direction = (rightVertex - leftVertex).head<3>().normalized();

    // create and push back a cylinder
    cylinders.push_back(Cylinder(leftVertex.head<3>(), rightVertex.head<3>(), direction, 0.05, VEC3(1, 0, 0)));
    spheres.push_back(Sphere(leftVertex.head<3>(), 0.05, VEC3(1, 0, 0)));
    spheres.push_back(Sphere(rightVertex.head<3>(), 0.05, VEC3(1, 0, 0)));
  }
}

//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
  string skeletonFilename("13.asf");
  string motionFilename("13_21.amc");

  // load up skeleton stuff
  skeleton = new Skeleton(skeletonFilename.c_str(), MOCAP_SCALE);
  skeleton->setBasePosture();
  displayer.LoadSkeleton(skeleton);

  // load up the motion
  motion = new Motion(motionFilename.c_str(), MOCAP_SCALE, skeleton);
  displayer.LoadMotion(motion);
  skeleton->setPosture(*(displayer.GetSkeletonMotion(0)->GetPosture(0)));

  setUpTexture(texturePath1);
  setUpTexture(texturePath2);
  setUpTexture(texturePath3);

  if (argc > 1 && atoi(argv[1]) == 1)
  {
    frame = 52;
    int x = 8 * frame;
    setSkeletonsToSpecifiedFrame(x);
    buildScene();

    char buffer[256];
    snprintf(buffer, 256, "./frames/frame.%04i.ppm", x / 8);
    renderImage(windowWidth, windowHeight, buffer);
    cout << "Rendered " + to_string(x / 8) + " frames" << endl;
  }
  else
  {
    // Note we're going 8 frames at a time, otherwise the animation is really slow.
    for (int x = 8 * frame; x < 8 * 300; x += 8)
    {
      setSkeletonsToSpecifiedFrame(x);
      buildScene();

      char buffer[256];
      snprintf(buffer, 256, "./frames/frame.%04i.ppm", x / 8);
      cout << "enter render" << endl;
      renderImage(windowWidth, windowHeight, buffer);
      cout << "Rendered " + to_string(x / 8) + " frames" << endl;
    }
  }

  return 0;
}
