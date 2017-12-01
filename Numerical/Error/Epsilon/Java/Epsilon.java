public class Epsilon{

    public static void main(String[] args){
        System.out.println("Double precision\tSingle precision");

        int count = 0;
        float d = 0.5f;
        double e = 0.5;

        while(1 != 1 + e) {

            if(1 != 1 + d){
                System.out.print("Iteration "+count+" = "+e+"\t");
                System.out.println("Iteration "+count+" = "+d);
                d = d / 2;
            }
            else{
                System.out.println("Iteration "+count+" = "+e);
            }

            e = e / 2;

            count++;
        }

    }

}
