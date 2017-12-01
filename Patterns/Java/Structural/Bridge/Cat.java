public class Cat extends Animal{

    public void jump(){
        movement = new Jump();

        System.out.print("The cat ");
        movement.move();
    }

    public void run(){
        movement = new Run();

        System.out.print("The cat ");
        movement.move();
    }

}
