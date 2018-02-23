turtles-own [ velocity ]

to setup
  ca
  reset-ticks

  crt 1 [
    setxy 0 16
    set color blue
    set heading 180
    set velocity 1
  ]

  ask n-of 5 patches [ set pcolor red ]

  ask patch 0 -9 [ set pcolor red ]
end

to go
  tick
  ask turtles [ move ]
end

to move
  let danger-ahead search-danger-ahead

  ifelse  any? danger-ahead [
    crash  danger-ahead
  ][
    advance
  ]
end

to advance
  fd velocity
  set velocity velocity * 1.4
end

to crash [danger-ahead]
  set velocity 0
  show "POW"
  let crash-patch min-one-of danger-ahead [distance myself]
  move-to crash-patch
end

to-report search-danger-ahead
  let distances n-values velocity [ x -> x + 1]
  let patches-ahead patch-set ( map patch-ahead distances )
  let red-patches-ahead patches-ahead with [pcolor = red]

  report red-patches-ahead
end
