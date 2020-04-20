from models import resnet, resstage, iresnet, resgroup, resgroupfix, iresgroupfix, iresgroup


def build_model(args):

    if args.arch == 'iresgroup':
        assert args.model_depth in [50, 101, 152]

        if args.model_depth == 50:
            model = iresgroup.iresgroup50(
                pretrained=args.pretrained,
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual,
                groups=args.groups)
        elif args.model_depth == 101:
            model = iresgroup.iresgroup101(
                pretrained=args.pretrained,
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual,
                groups=args.groups)
        elif args.model_depth == 152:
            model = iresgroup.iresgroup152(
                pretrained=args.pretrained,
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual,
                groups=args.groups)

    if args.arch == 'iresgroupfix':
        assert args.model_depth in [50, 101, 152]

        if args.model_depth == 50:
            model = iresgroupfix.iresgroupfix50(
                pretrained=args.pretrained,
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual,
                groups=args.groups)
        elif args.model_depth == 101:
            model = iresgroupfix.iresgroupfix101(
                pretrained=args.pretrained,
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual,
                groups=args.groups)
        elif args.model_depth == 152:
            model = iresgroupfix.iresgroupfix152(
                pretrained=args.pretrained,
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual,
                groups=args.groups)

    if args.arch == 'resgroupfix':
        assert args.model_depth in [50, 101, 152]

        if args.model_depth == 50:
            model = resgroupfix.resgroupfix50(
                pretrained=args.pretrained,
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual,
                groups=args.groups)
        elif args.model_depth == 101:
            model = resgroupfix.resgroupfix101(
                pretrained=args.pretrained,
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual,
                groups=args.groups)
        elif args.model_depth == 152:
            model = resgroupfix.resgroupfix152(
                pretrained=args.pretrained,
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual,
                groups=args.groups)

    if args.arch == 'resgroup':
        assert args.model_depth in [50, 101, 152]

        if args.model_depth == 50:
            model = resgroup.resgroup50(
                pretrained=args.pretrained,
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual,
                groups=args.groups)
        elif args.model_depth == 101:
            model = resgroup.resgroup101(
                pretrained=args.pretrained,
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual,
                groups=args.groups)
        elif args.model_depth == 152:
            model = resgroup.resgroup152(
                pretrained=args.pretrained,
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual,
                groups=args.groups)

    if args.arch == 'iresnet':
        assert args.model_depth in [18, 34, 50, 101, 152, 200, 302, 404, 1001]

        if args.model_depth == 18:
            model = iresnet.iresnet18(
                pretrained=args.pretrained,
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)
        elif args.model_depth == 34:
            model = iresnet.iresnet34(
                pretrained=args.pretrained,
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)
        elif args.model_depth == 50:
            model = iresnet.iresnet50(
                pretrained=args.pretrained,
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)
        elif args.model_depth == 101:
            model = iresnet.iresnet101(
                pretrained=args.pretrained,
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)
        elif args.model_depth == 152:
            model = iresnet.iresnet152(
                pretrained=args.pretrained,
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)
        elif args.model_depth == 200:
            model = iresnet.iresnet200(
                pretrained=args.pretrained,
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)
        elif args.model_depth == 302:
            model = iresnet.iresnet302(
                pretrained=args.pretrained,
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)
        elif args.model_depth == 404:
            model = iresnet.iresnet404(
                pretrained=args.pretrained,
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)
        elif args.model_depth == 1001:
            model = iresnet.iresnet1001(
                pretrained=args.pretrained,
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)

    if args.arch == 'resstage':
        assert args.model_depth in [18, 34, 50, 101, 152, 200]

        if args.model_depth == 18:
            model = resstage.resstage18(
                pretrained=args.pretrained,
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)
        elif args.model_depth == 34:
            model = resstage.resstage34(
                pretrained=args.pretrained,
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)
        elif args.model_depth == 50:
            model = resstage.resstage50(
                pretrained=args.pretrained,
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)
        elif args.model_depth == 101:
            model = resstage.resstage101(
                pretrained=args.pretrained,
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)
        elif args.model_depth == 152:
            model = resstage.resstage152(
                pretrained=args.pretrained,
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)
        elif args.model_depth == 200:
            model = resstage.resstage200(
                pretrained=args.pretrained,
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)

    if args.arch == 'resnet':
        assert args.model_depth in [18, 34, 50, 101, 152, 200]

        if args.model_depth == 18:
            model = resnet.resnet18(
                pretrained=args.pretrained,
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)
        elif args.model_depth == 34:
            model = resnet.resnet34(
                pretrained=args.pretrained,
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)
        elif args.model_depth == 50:
            model = resnet.resnet50(
                pretrained=args.pretrained,
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)
        elif args.model_depth == 101:
            model = resnet.resnet101(
                pretrained=args.pretrained,
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)
        elif args.model_depth == 152:
            model = resnet.resnet152(
                pretrained=args.pretrained,
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)
        elif args.model_depth == 200:
            model = resnet.resnet200(
                pretrained=args.pretrained,
                num_classes=args.n_classes,
                zero_init_residual=args.zero_init_residual)

    return model
