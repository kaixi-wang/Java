public class Main {
    public static void main(String args[]) {
        ImageDisplay ui = new ImageDisplay(); // create and display GUI   
        ui.Initialize(args);
        gameLoop(ui, args); // start the game loop
    }

    static void gameLoop(ImageDisplay ui, String args[]) {
        while (true) {
            try{
                ui.update();
                Thread.sleep(10);
            } catch (Exception e) {
                System.out.println(e.toString());
            }
		}
    }
}