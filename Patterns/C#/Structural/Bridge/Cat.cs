using System;

public class Cat : Animal{

    public override void jump(){

        movement = new Jump();

        Console.Write("The cat ");
        movement.move();
    }

    public override void run(){

        movement = new Run();

        Console.Write("The cat ");
        movement.move();
    }

}
