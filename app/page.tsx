import Link from "next/link";

export default function Page() {
    return (
        <div className="flex min-h-screen flex-col items-center justify-center p-24 bg-background">
            <div className="z-10 max-w-5xl w-full items-center justify-between font-mono text-sm lg:flex pb-12">
                <p className="fixed left-0 top-0 flex w-full justify-center border-b border-gray-300 bg-gradient-to-b from-zinc-200 pb-6 pt-8 backdrop-blur-2xl dark:border-neutral-800 dark:bg-zinc-800/30 dark:from-inherit lg:static lg:w-auto  lg:rounded-xl lg:border lg:bg-gray-200 lg:p-4 lg:dark:bg-zinc-800/30">
                    Matcher&nbsp;
                    <code className="font-mono font-bold">OAuth Provider</code>
                </p>
            </div>

            <div className="relative flex place-items-center mb-12">
                <h1 className="text-6xl font-extrabold tracking-tight lg:text-8xl bg-clip-text text-transparent bg-gradient-to-r from-primary to-purple-600">
                    Matcher
                </h1>
            </div>

            <div className="mb-32 grid text-center lg:max-w-5xl lg:w-full lg:mb-0 lg:grid-cols-2 lg:text-left gap-8">
                <Link
                    href="/api/auth/signin"
                    className="group rounded-lg border border-transparent px-5 py-4 transition-colors hover:border-gray-300 hover:bg-gray-100 hover:dark:border-neutral-700 hover:dark:bg-neutral-800/30"
                >
                    <h2 className={`mb-3 text-2xl font-semibold`}>
                        Login{" "}
                        <span className="inline-block transition-transform group-hover:translate-x-1 motion-reduce:transform-none">
                            -&gt;
                        </span>
                    </h2>
                    <p className={`m-0 max-w-[30ch] text-sm opacity-50`}>
                        Sign in to your account and manage your profile.
                    </p>
                </Link>

                <Link
                    href="/api/auth/signin?type=register"
                    className="group rounded-lg border border-transparent px-5 py-4 transition-colors hover:border-gray-300 hover:bg-gray-100 hover:dark:border-neutral-700 hover:dark:bg-neutral-800/30"
                >
                    <h2 className={`mb-3 text-2xl font-semibold`}>
                        Sign Up{" "}
                        <span className="inline-block transition-transform group-hover:translate-x-1 motion-reduce:transform-none">
                            -&gt;
                        </span>
                    </h2>
                    <p className={`m-0 max-w-[30ch] text-sm opacity-50`}>
                        Create a new account and start matching.
                    </p>
                </Link>
            </div>
        </div>
    );
}