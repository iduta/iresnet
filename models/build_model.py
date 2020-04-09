from models import resnet, resstage, iresnet, resgroup, resgroupfix, iresgroup, iresgroupfix


def build_model(args):

    if args.arch == 'iresgroupfix':

        if args.model_depth == 18:
            model = iresgroupfix.iresgroupfix50(
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual,
                groups=args.groups)
        elif args.model_depth == 34:
            model = iresgroupfix.iresgroupfix101(
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual,
                groups=args.groups)
        elif args.model_depth == 50:
            model = iresgroupfix.iresgroupfix152(
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual,
                groups=args.groups)

    if args.arch == 'iresgroup':

        if args.model_depth == 18:
            model = iresgroup.iresgroup50(
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual,
                groups=args.groups)
        elif args.model_depth == 34:
            model = iresgroup.iresgroup101(
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual,
                groups=args.groups)
        elif args.model_depth == 50:
            model = iresgroup.iresgroup152(
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual,
                groups=args.groups)

    if args.arch == 'resgroupfix':

        if args.model_depth == 18:
            model = resgroupfix.resgroupfix50(
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual,
                groups=args.groups)
        elif args.model_depth == 34:
            model = resgroupfix.resgroupfix101(
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual,
                groups=args.groups)
        elif args.model_depth == 50:
            model = resgroupfix.resgroupfix152(
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual,
                groups=args.groups)

    if args.arch == 'resgroup':

        if args.model_depth == 18:
            model = resgroup.resgroup50(
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual,
                groups=args.groups)
        elif args.model_depth == 34:
            model = resgroup.resgroup101(
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual,
                groups=args.groups)
        elif args.model_depth == 50:
            model = resgroup.resgroup152(
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual,
                groups=args.groups)

    if args.arch == 'iresnet':

        if args.model_depth == 18:
            model = iresnet.iresnet18(
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)
        elif args.model_depth == 34:
            model = iresnet.iresnet34(
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)
        elif args.model_depth == 50:
            model = iresnet.iresnet50(
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)
        elif args.model_depth == 101:
            model = iresnet.iresnet101(
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)
        elif args.model_depth == 152:
            model = iresnet.iresnet152(
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)
        elif args.model_depth == 200:
            model = iresnet.iresnet200(
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)
        elif args.model_depth == 302:
            model = iresnet.iresnet302(
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)
        elif args.model_depth == 404:
            model = iresnet.iresnet404(
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)
        elif args.model_depth == 1001:
            model = iresnet.iresnet1001(
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)

    if args.arch == 'resstage':

        if args.model_depth == 18:
            model = resstage.resstage18(
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)
        elif args.model_depth == 34:
            model = resstage.resstage34(
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)
        elif args.model_depth == 50:
            model = resstage.resstage50(
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)
        elif args.model_depth == 101:
            model = resstage.resstage101(
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)
        elif args.model_depth == 152:
            model = resstage.resstage152(
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)
        elif args.model_depth == 200:
            model = resstage.resstage200(
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)

    if args.arch == 'resnet':

        if args.model_depth == 18:
            model = resnet.resnet18(
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)
        elif args.model_depth == 34:
            model = resnet.resnet34(
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)
        elif args.model_depth == 50:
            model = resnet.resnet50(
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)
        elif args.model_depth == 101:
            model = resnet.resnet101(
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)
        elif args.model_depth == 152:
            model = resnet.resnet152(
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)
        elif args.model_depth == 200:
            model = resnet.resnet200(
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)

    return model