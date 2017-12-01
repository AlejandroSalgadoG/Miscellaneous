public class MainBridge{

    public static void main(String[] args){

        Animal dog = new Dog();
        Animal cat = new Cat();

        dog.jump();
        dog.run();

        System.out.println();

        cat.jump();
        cat.run();

    }

}
