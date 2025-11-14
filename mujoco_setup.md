# Eureka on MuJoCo Environments

To fully test the generality of Eureka with regard to code syntax and physics simulation, we also evaluate Eureka on the OpenAI Gym Humanoid environment (Brockman et al., 2016) implemented using MuJoCo (Todorov et al., 2012).

## Changes from Isaac Gym Implementation

We make no changes to the prompts in App. A except the formatting tip that instructs the LLM to use numpy array instead of pytorch tensor when performing matrix operations in the generated program.

## Observation Code Differences

The observation code for Eureka context is displayed in Example 1 below. As shown, compared to the observation code for the Isaac Gym Humanoid task (Example 1 in App. D), this variant conveys the observation information in a very different manner:

- A commented block reveals the state and action spaces
- The variables themselves have to be indexed from a monolithic observation array

## Results

The comparison against the official human-written reward function is in Tab. 4. Despite vastly different observation space and code syntax, Eureka remains effective and generates reward functions that outperform the official human reward function.

## Example 1

Mujoco Humanoid environment observation code given to Eureka.

