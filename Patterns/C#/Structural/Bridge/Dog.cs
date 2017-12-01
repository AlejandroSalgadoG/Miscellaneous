using System;

public class Dog : Animal{

    public override void jump(){

        movement = new Jump();

        Console.Write("The dog ");
        movement.move();
    }

    public override void run(){

        movement = new Run();

        Console.Write("The dog ");
        movement.move();
    }

}
