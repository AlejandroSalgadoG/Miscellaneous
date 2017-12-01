public class MainComposite{

    public static void main(String[] args){
        Boss boss1 = new Boss("Boss1");

        boss1.add(new Employee("Employee1"));

        Boss boss2 = new Boss("Boss1-2");

        boss2.add( new Employee("Employee1-2") );
        boss2.add( new Assistant("Assistant1-2") );

        boss1.add(boss2);

        boss1.add( new Assistant("Assistant1") );

        boss1.displayStructure();
    }

}
