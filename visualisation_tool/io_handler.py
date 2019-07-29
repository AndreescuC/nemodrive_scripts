from argparse import ArgumentParser


def build_parser():
    parser = ArgumentParser()

    parser.add_argument("--frame_delay", type=float, default=0.01, help="Delay between frames")
    parser.add_argument(
        "--particle_filter",
        dest="particle_filter",
        default=False,
        help="Computes particles at each state and evaluates them based on segmentation frames => higher frame delay"
    )
    parser.add_argument(
        "--heat_map",
        dest="heat_map",
        default=False,
        help="Displays a heat map at each state based on particle weights; used only with particle_filter"
    )

    return parser


def get_parameters():
    parser = build_parser()
    args = parser.parse_args()

    params = {
        'frame_delay': args.frame_delay,
        'particle_filter': args.particle_filter,
        'heat_map': args.heat_map,
    }

    return params
