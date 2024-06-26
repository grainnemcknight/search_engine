import React, { FC } from "react";

export const Footer: FC = () => {
  return (
    <div className="text-center flex flex-col items-center text-xs text-zinc-700 gap-1">
      <div className="text-zinc-400">
        Answer generated by large language models, please double check for
        correctness.
      </div>
    </div>
  );
};
