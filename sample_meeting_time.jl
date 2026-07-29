"""
sample_meeting_time(pi_0, skernel, ckernel; lag = 1, maxit = 1_000_000)

Sample meeting time of chains that start from `pi_0`,
then one chain moves according to `skernel` for `lag` steps,
then both chains move according to `ckernel` until they meet.

# Arguments
- `pi_0`: a function that takes zero arguments and returns a state of the chain
- `skernel`: a function that takes a state of the chain and return another state
- `ckernel`: a function that takes two states and returns two states, as well as a boolean indicator of meeting
- `lag::Integer=1`: time lag between the chains
- `maxit::Integer=1_000_000`: maximum number of steps to perform before giving up

"""
function sample_meeting_time(pi_0, skernel, ckernel; lag = 1, maxit = 1_000_000)
  ## wall-clock time should be measured
  # elapsed=Inf
  # start = now()
  time = 0
  state1 = pi_0()
  state2 = pi_0()
  for step in 1:lag
      time = time + 1
      state1 = skernel(state1)
  end
  meetingtime = Inf
  while time < maxit && isinf(meetingtime)        
      time = time + 1
      state1, state2, identical = ckernel(state1, state2)
      if identical
          meetingtime = time
          # elapsed = now() - start
      end
  end  
  return meetingtime #, elapsed
end 

"""
Obtain TV upper bounds from meeting times
"""
function tv_upper_bound(meetingtimes, lag, t)
  return mean(max.(0, ceil.((meetingtimes .- lag .- t) ./ lag)))
end
