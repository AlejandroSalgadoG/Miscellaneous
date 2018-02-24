breed [ farmers farmer ]
breed [ bandits bandit ]
breed [ soldiers soldier ]

to setup
  ca
  reset-ticks

  create-farmers 1 [
    setxy random-xcor random-ycor
    set color orange
    ;set shape "person farmer"
  ]

;  crt 5 [
;    setxy random-xcor random-ycor
;    set shape "person soldier"
;  ]

  create-bandits 1 [
    setxy random-xcor random-ycor
    set color blue
    ;set shape "person lumberjack"
  ]
end

to go
  tick

  ask farmers [
    let closest-bandit min-one-of bandits [ distance myself ]

    ifelse distance closest-bandit > 5 [
      set heading random 360
      fd 1
    ][
      face closest-bandit
      rt 180
      fd 1
    ]
  ]

  ask bandits [
    let closest-farmer min-one-of farmers [ distance myself ]
    face closest-farmer
    fd 1
  ]
end
