/*
 * Author     :  (DSP Group, E&E Eng, US)
 * Created on :
 * Copyright  : University of Stellenbosch, all rights retained
 */

// patrec headers
#include "prlite_logging.hpp" // initLogging
#include "prlite_testing.hpp"

// emdw headers
#include "emdw.hpp"
#include "discretetable.hpp"
#include "combinations.hpp"

// standard headers
#include <iostream> // cout, endl, flush, cin, cerr
#include <cctype>   // toupper
#include <string>   // string
#include <memory>
#include <set>
#include <map>
#include <algorithm>
#include <limits>
#include <random>

#include "prlite_genmat.hpp" // For Guassian Factors
#include "sqrtmvg.hpp"

#include <fstream> //Reading and writing to and from textfiles
#include <sstream>

// In order to use the built in Inference packages
// standard library headers
#include <vector> // vector

// emdw headers
#include "clustergraph.hpp"
#include "lbp_cg.hpp"
#include "lbu_cg.hpp"
#include "messagequeue.hpp"

using namespace std;
using namespace emdw;

// ##################################################################
//  Some example code. To compile this, go to the emdw/build
//  directory and do a:
//  cmake ../; make -j7 example
//  To run this while in the build directory, do a:
//  src/pmr/example
//
//  For your own stuff, make a copy of this one to start with. Then
//  edit the CMakeLists.txt (also in this directory) by adding your
//  new target in the same way as this example.
// ##################################################################

// Function to read ball positions from a text file and store them in a vector of pairs
std::vector<std::pair<double, double>> readPositions(const std::string &filepath)
{ // Input: filepath → the path to the text file containing x,y coordinates

  // Create an empty vector to store the positions
  std::vector<std::pair<double, double>> positions;

  // Open the file at the given path for reading
  std::ifstream infile(filepath);

  // Check if the file opened successfully
  if (!infile.is_open())
  {
    // If the file could not be opened, print an error message
    std::cerr << "Error: Could not open file " << filepath << std::endl;
    // Return the empty vector because we cannot read data
    return positions;
  }

  // String to hold each line of the file as we read it
  std::string line;

  // Read the file line by line
  while (std::getline(infile, line))
  {
    // If the line is empty, skip it and continue to the next line
    if (line.empty())
      continue;

    // Create a stringstream from the line so we can split it by commas
    std::stringstream ss(line);

    // Strings to hold the x and y values as text
    std::string x_str, y_str;

    // Split the line into x and y using the comma as a delimiter
    // std::getline(ss, x_str, ',') → reads characters until ',' into x_str
    // std::getline(ss, y_str) → reads the remaining characters into y_str
    if (std::getline(ss, x_str, ',') && std::getline(ss, y_str))
    {

      // Convert x from string to double
      double x = std::stod(x_str);
      // Convert y from string to double
      double y = std::stod(y_str);

      // Add the pair (x, y) to the vector of positions
      positions.emplace_back(x, y);
    }
  }

  // Close the file after reading
  infile.close();

  // Return the vector containing all positions
  return positions; // Output: vector of pairs of doubles, each pair representing (x, y) positions
}

// Function to count the number of rows in a text file
int countFrames(const std::string &filepath)
{

  std::ifstream infile(filepath);

  if (!infile.is_open())
  {
    std::cerr << "Error: Could not open file " << filepath << std::endl;
    return 0;
  }

  std::string line;
  int frames = 0;
  while (std::getline(infile, line))
  {
    if (!line.empty())
      frames++; // ignore empty lines
  }

  infile.close();
  return frames;
}

static void outputToFile(double x, double y)
{

  std::string outputFile = "/home/alex/devel/emdw/build/outputs.txt";

  // Open file in append mode
  std::ofstream outfile(outputFile, std::ios::app);

  // Check if file opened
  if (!outfile.is_open())
  {
    std::cerr << "Error: Could not open file " << outputFile << std::endl;
    return;
  }

  // Write x and y separated by a comma, then go to the next line
  outfile << x << "," << y << std::endl;
}

int main(int, char *argv[])
{

  // NOTE: this activates logging and unit tests
  initLogging(argv[0]);
  prlite::TestCase::runAllTests();

  std::ofstream outfile("/home/alex/devel/emdw/build/outputs.txt", std::ios::trunc);
  outfile.close();

  try
  {

    //*********************************************************
    // Some random generator seeding. Just keep this as is
    //*********************************************************

    unsigned seedVal = emdw::randomEngine.getSeedVal();
    cout << seedVal << endl;
    emdw::randomEngine.setSeedVal(seedVal);

    //*********************************************************
    // Predefine some types and constants
    //*********************************************************
    typedef SqrtMVG SG; // Shorthand for SqrtMVG

    std::vector<std::string> filePaths = {
        "/home/alex/devel/emdw/build/alg1.txt",
        "/home/alex/devel/emdw/build/alg2.txt",
        "/home/alex/devel/emdw/build/alg3.txt",
        "/home/alex/devel/emdw/build/alg4.txt",
        "/home/alex/devel/emdw/build/alg5.txt"};

    int frames = 0; // Number of framess which are being tracked, fps*video length

    for (int i = 0; i < 5; i++)
    {

      int temp = countFrames(filePaths[i]);

      if (temp > frames)
      {
        frames = temp;
      }
    }

    cout << "Number of Frames: " << frames << "\n";

    // calling function which populates posAlg1 with all the x,y values from the textfiles
    std::vector<std::pair<double, double>> posAlg1 = readPositions(filePaths[0]);
    std::vector<std::pair<double, double>> posAlg2 = readPositions(filePaths[1]);
    std::vector<std::pair<double, double>> posAlg3 = readPositions(filePaths[2]);
    std::vector<std::pair<double, double>> posAlg4 = readPositions(filePaths[3]);
    std::vector<std::pair<double, double>> posAlg5 = readPositions(filePaths[4]);

    std::vector<double> posAlg1x, posAlg1y, posAlg2x, posAlg2y, posAlg3x, posAlg3y, posAlg4x, posAlg4y, posAlg5x, posAlg5y; // Creating arrays (vectors) from all the x and y positions

    for (int i = 0; i < frames; i++)
    {
      posAlg1x.push_back(posAlg1[i].first);
      posAlg1y.push_back(posAlg1[i].second);
      posAlg2x.push_back(posAlg2[i].first);
      posAlg2y.push_back(posAlg2[i].second);
      posAlg3x.push_back(posAlg3[i].first);
      posAlg3y.push_back(posAlg3[i].second);
      posAlg4x.push_back(posAlg4[i].first);
      posAlg4y.push_back(posAlg4[i].second);
      posAlg5x.push_back(posAlg5[i].first);
      posAlg5y.push_back(posAlg5[i].second);
    }

    // These need to be refined per algorithm
    prlite::RowMatrix<double> cov_Ipa1(2, 2); // covariance matrix 1 (I have set these as constant, but may need to modify as I get furhter for more refinement)
    cov_Ipa1(0, 0) = 0.2;
    cov_Ipa1(1, 1) = 0.2;     // sets the confidence I have in the algorihtms, lower the more confident
    cov_Ipa1(0, 1) = cov_Ipa1(1, 0) = 0.0;

    prlite::RowMatrix<double> cov_Ipa2(2, 2); // covariance matrix 1 (I have set these as constant, but may need to modify as I get furhter for more refinement)
    cov_Ipa2(0, 0) = 0.2;
    cov_Ipa2(1, 1) = 0.2;     // sets the confidence I have in the algorihtms, lower the more confident
    cov_Ipa2(0, 1) = cov_Ipa2(1, 0) = 0.0;

    prlite::RowMatrix<double> cov_Ipa3(2, 2); // covariance matrix 1 (I have set these as constant, but may need to modify as I get furhter for more refinement)
    cov_Ipa3(0, 0) = 0.2;
    cov_Ipa3(1, 1) = 0.2;     // sets the confidence I have in the algorihtms, lower the more confident
    cov_Ipa3(0, 1) = cov_Ipa3(1, 0) = 0.0;

    prlite::RowMatrix<double> cov_Ipa4(2, 2); // covariance matrix 1 (I have set these as constant, but may need to modify as I get furhter for more refinement)
    cov_Ipa4(0, 0) = 0.2;
    cov_Ipa4(1, 1) = 0.2;     // sets the confidence I have in the algorihtms, lower the more confident
    cov_Ipa4(0, 1) = cov_Ipa4(1, 0) = 0.0;

    prlite::RowMatrix<double> cov_Ipa5(2, 2); // covariance matrix 1 (I have set these as constant, but may need to modify as I get furhter for more refinement)
    cov_Ipa5(0, 0) = 0.2;
    cov_Ipa5(1, 1) = 0.2;     // sets the confidence I have in the algorihtms, lower the more confident
    cov_Ipa5(0, 1) = cov_Ipa5(1, 0) = 0.0;

    prlite::ColVector<double> mean_trans(4); // mean vector for the transition factor
    mean_trans[0] = mean_trans[1] = mean_trans[2] = mean_trans[3] = 0;

    // These need to be refined
    //  Transition factor: (x_prev, y_prev) -> (x, y)
    prlite::RowMatrix<double> cov_trans(4, 4);
    for (int d = 0; d < 4; d++)
    {
      for (int e = 0; e < 4; e++)
        cov_trans(d, e) = 0.0; // zero everything
      cov_trans(d, d) = 0.2;   // set diagonal // sets the confience I have in the model, lower the more confident
    } // This can be adjusted as needed

    // Start the for loop to iterate through the frames, However might do 2 forms of inference BU and BP

    map<RVIdType, AnyType> obvs;

    unsigned int x_prev = 0;
    unsigned int y_prev = 1;

    for (int i = 0; i < frames; i++)
    {

      vector<rcptr<Factor>> factors;

      unsigned int x = 2 * i;
      unsigned int y = 2 * i + 1;

      prlite::ColVector<double> mn1(2); // mean vector 1
      mn1[0] = posAlg1x[i];
      mn1[1] = posAlg1y[i];

      bool ignoreAlg1 = (mn1[0] == -1 && mn1[1] == -1); // to see if we should ignore this algorithm ie. if x,y = -1,-1;

      prlite::ColVector<double> mn2(2); // Mean vector 2
      mn2[0] = posAlg2x[i];
      mn2[1] = posAlg2y[i];

      bool ignoreAlg2 = (mn2[0] == -1 && mn2[1] == -1); // to see if we should ignore this algorithm ie. if x,y = -1,-1;

      prlite::ColVector<double> mn3(2); // Mean vector 3
      mn3[0] = posAlg3x[i];
      mn3[1] = posAlg3y[i];

      bool ignoreAlg3 = (mn3[0] == -1 && mn3[1] == -1); // to see if we should ignore this algorithm ie. if x,y = -1,-1;

      prlite::ColVector<double> mn4(2); // Mean vector 4
      mn4[0] = posAlg4x[i];
      mn4[1] = posAlg4y[i];

      bool ignoreAlg4 = (mn4[0] == -1 && mn4[1] == -1); // to see if we should ignore this algorithm ie. if x,y = -1,-1;

      prlite::ColVector<double> mn5(2); // Mean vector 5
      mn5[0] = posAlg5x[i];
      mn5[1] = posAlg5y[i];

      bool ignoreAlg5 = (mn5[0] == -1 && mn5[1] == -1); // to see if we should ignore this algorithm ie. if x,y = -1,-1;

      if (!ignoreAlg1)
      {
        std::cout << "Frame " << i << " Alg1: (" << mn1[0] << "," << mn1[1] << ") ignore=" << ignoreAlg1 << "\n";
        rcptr<Factor> alg_1(uniqptr<SG>(new SG({x, y}, mn1, cov_Ipa1))); // Setting up the Guassian factor for IPA1
        factors.push_back(alg_1);
      }

      if (!ignoreAlg2)
      {
        std::cout << "Frame " << i << " Alg2: (" << mn2[0] << "," << mn2[1] << ") ignore=" << ignoreAlg2 << "\n";
        rcptr<Factor> alg_2(uniqptr<SG>(new SG({x, y}, mn2, cov_Ipa2))); // Setting up the Guassian factor for IPA2
        factors.push_back(alg_2);
      }

      if (!ignoreAlg3)
      {
        std::cout << "Frame " << i << " Alg3: (" << mn3[0] << "," << mn3[1] << ") ignore=" << ignoreAlg3 << "\n";
        rcptr<Factor> alg_3(uniqptr<SG>(new SG({x, y}, mn3, cov_Ipa3))); // Setting up the Guassian factor for IPA3
        factors.push_back(alg_3);
      }

      if (!ignoreAlg4)
      {
        std::cout << "Frame " << i << " Alg4: (" << mn4[0] << "," << mn4[1] << ") ignore=" << ignoreAlg4 << "\n";
        rcptr<Factor> alg_4(uniqptr<SG>(new SG({x, y}, mn4, cov_Ipa4))); // Setting up the Guassian factor for IPA4
        factors.push_back(alg_4);
      }
      if (!ignoreAlg5)
      {
        std::cout << "Frame " << i << " Alg5: (" << mn5[0] << "," << mn5[1] << ") ignore=" << ignoreAlg5 << "\n";
        rcptr<Factor> alg_5(uniqptr<SG>(new SG({x, y}, mn5, cov_Ipa5))); // Setting up the Guassian factor for IPA5
        factors.push_back(alg_5);
      }
      if (i != 0)
      {
        rcptr<Factor> trans(new SqrtMVG({x_prev, y_prev, x, y}, mean_trans, cov_trans)); // Transition factor connecting the previous frame's (x_prev, y_prev) to the current frame's (x, y)
        factors.push_back(trans);                                                        // Create a vector of factors for the subsequent frames
      }
      ClusterGraph cg(ClusterGraph::LTRIP, factors, obvs); // Create a cluster graph with the factors

      map<Idx2z, rcptr<Factor>> msgs; // Calibrate via loopy belief propagation (Shafer–Shenoy)
      MessageQueue msgQ;
      unsigned nMsgs = loopyBP_CG(cg, msgs, msgQ);
      cout << "Sent " << nMsgs << " messages before convergence\n";

      rcptr<Factor> qPtr = queryLBP_CG(cg, msgs, {x, y})->normalize(); // Query the (x,y) belief for this frame

      auto qPtr_Sqrt = static_cast<SqrtMVG *>(qPtr.get());

      cout << "LBP belief mean for frame " << (i + 1) << ": " << qPtr_Sqrt->getMean() << endl;

      // need to calculate the usual delta between each frame to account for the change in speed etc

      auto mean = qPtr_Sqrt->getMean();
      if (i == 0)
      {
        mean_trans[0] = mean[0];
        mean_trans[1] = mean[1];
        mean_trans[2] = mean[0];
        mean_trans[3] = mean[1];
      }
      else
      {
        double deltaX = mean[0] - mean_trans[0];
        double deltaY = mean[1] - mean_trans[1];

        mean_trans[0] = mean[0];          // x from the prev frame
        mean_trans[1] = mean[1];          // y from the prev frame
        mean_trans[2] = mean[0] + deltaX; // expected position (for the second observation) based on this frame and the last frame
        mean_trans[3] = mean[1] + deltaY; // expected position (for the second observation) based on this frame and the last frame
        // These are for the transistion factor, which is used to get the next state based on the previous state
      }

      outputToFile(mean[0], mean[1]);
      x_prev = x; // Update previous x and y for the next iteration
      y_prev = y;
    }

    return 0; // tell the world that all is fine
  } // try

  catch (char msg[])
  {
    cerr << msg << endl;
  } // catch

  // catch (char const* msg) {
  //   cerr << msg << endl;
  // } // catch

  catch (const string &msg)
  {
    cerr << msg << endl;
    throw;
  } // catch

  catch (const exception &e)
  {
    cerr << "Unhandled exception: " << e.what() << endl;
    throw e;
  } // catch

  catch (...)
  {
    cerr << "An unknown exception / error occurred\n";
    throw;
  } // catch

} // main
